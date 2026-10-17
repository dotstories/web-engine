import type { Graph } from "../gpu/graph";
import { runAhead } from "../gpu/run-ahead";
import { attention } from "../kernels/attention";
import { positions, ropeTables } from "../kernels/rope";
import { layerSumWgsl, layerSumWorkgroups, shortAttentionParams, shortAttentionWgsl, shortAttentionWorkgroups } from "../kernels/text";
import { block, fusionBlock, KREA_BLOCK, type AttentionMlpWeights, type BlockConfig, type BlockWeights } from "./block";
import { add, gelu, linear, noMod, norm, type Linear } from "./ops";

/**
 * Krea 2's diffusion transformer (ComfyUI comfy/ldm/krea2/model.py SingleStreamDiT), parts around the blocks:
 * - time:  t = tmlp(sincos(1000 · timestep)), tvec = tproj(t) = Linear(GELU(t)) → 6 × hidden for every block
 * - text:  12 text-encoder layers per token → 2 blocks across the layers → 12→1 projector → 2 blocks
 *          across the tokens → txtmlp (norm, Linear, GELU, Linear) → hidden
 * - image: 2×2 patches of the 16-channel latent → Linear (first) → hidden
 * - all tokens [text, image] through the blocks, then last: (1 + scale) · norm(x) + shift, Linear → 64
 */
export const KREA_TEXT: BlockConfig = { hidden: 2560, mlp: 6912, heads: 20, kvHeads: 20, headDim: 128, eps: 1e-5 };
export const TEXT_LAYERS = 12;
export const LATENT_CHANNELS = 16;
export const TIME_DIM = 256;

export type TransformerWeights = {
  first: Linear;
  tmlp0: Linear;
  tmlp2: Linear;
  tproj: Linear;
  layerwise: AttentionMlpWeights[];
  projector: GPUBuffer; // [12]
  refiner: AttentionMlpWeights[];
  txtNorm: GPUBuffer;
  txt1: Linear;
  txt3: Linear;
  blocks: BlockWeights[];
  lastNorm: GPUBuffer;
  lastScale: GPUBuffer; // modulation table row 0
  lastShift: GPUBuffer; // row 1
  lastLinear: Linear;
};

/** Flux/ComfyUI timestep_embedding: [cos(1000·t·f), sin(1000·t·f)], f = 10000^(-i / 128). */
export function timestepEmbedding(timestep: number, dim = TIME_DIM): Float32Array<ArrayBuffer> {
  const half = dim / 2;
  const out = new Float32Array(dim);
  for (let i = 0; i < half; i++) {
    const arg = 1000 * timestep * Math.exp((-Math.log(10000) * i) / half);
    out[i] = Math.cos(arg);
    out[half + i] = Math.sin(arg);
  }
  return out;
}

/** Token grid of a latent: 2×2 patches, odd sizes padded up (ComfyUI pad_to_patch_size). */
export const patchGrid = (height: number, width: number) => ({ rows: Math.ceil(height / 2), cols: Math.ceil(width / 2) });

/**
 * Latent [16, H, W] → tokens [rows · cols, 64], features in (channel, row in patch, column in patch) order.
 * Odd H or W is padded circularly (wraps to row/column 0), as ComfyUI does: the Kit's 1928 px is 241 columns.
 */
export function patchify(latent: Float32Array, height: number, width: number, channels = LATENT_CHANNELS): Float32Array<ArrayBuffer> {
  const { rows, cols } = patchGrid(height, width);
  const out = new Float32Array(rows * cols * channels * 4);
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++)
      for (let ch = 0; ch < channels; ch++)
        for (let pr = 0; pr < 2; pr++)
          for (let pc = 0; pc < 2; pc++)
            out[(r * cols + c) * channels * 4 + ch * 4 + pr * 2 + pc] = latent[(ch * height + ((r * 2 + pr) % height)) * width + ((c * 2 + pc) % width)];
  return out;
}

/** The inverse of patchify; the padding is cropped back off. */
export function unpatchify(tokens: Float32Array, height: number, width: number, channels = LATENT_CHANNELS): Float32Array<ArrayBuffer> {
  const { rows, cols } = patchGrid(height, width);
  const out = new Float32Array(channels * height * width);
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++)
      for (let ch = 0; ch < channels; ch++)
        for (let pr = 0; pr < 2 && r * 2 + pr < height; pr++)
          for (let pc = 0; pc < 2 && c * 2 + pc < width; pc++)
            out[(ch * height + r * 2 + pr) * width + c * 2 + pc] = tokens[(r * cols + c) * channels * 4 + ch * 4 + pr * 2 + pc];
  return out;
}

/** t [hidden] and tvec [6 × hidden] for one timestep. */
export function time(graph: Graph, w: TransformerWeights, timestep: number) {
  const emb = graph.buffer(timestepEmbedding(timestep));
  const t = linear(graph, gelu(graph, linear(graph, emb, w.tmlp0, 1), w.tmlp0.n), w.tmlp2, 1);
  const tvec = linear(graph, gelu(graph, t, w.tmlp2.n), w.tproj, 1);
  return { t, tvec };
}

/** Text hidden states [tokens, 12 · 2560] → [tokens, hidden]. Runs once per prompt. */
export function text(graph: Graph, w: TransformerWeights, context: GPUBuffer, tokens: number) {
  const cfg = KREA_TEXT;
  const shape = { batch: tokens, seq: TEXT_LAYERS, heads: cfg.heads, dim: cfg.headDim };
  const acrossLayers = (q: GPUBuffer, k: GPUBuffer, v: GPUBuffer) => {
    const out = graph.buffer(tokens * TEXT_LAYERS * cfg.hidden * 4);
    graph.run(shortAttentionWgsl, [graph.buffer(shortAttentionParams(shape)), q, k, v, out], shortAttentionWorkgroups(shape));
    return out;
  };
  const acrossTokens = (q: GPUBuffer, k: GPUBuffer, v: GPUBuffer) =>
    attention(graph, q, k, v, { tokens, heads: cfg.heads, kvHeads: cfg.kvHeads, dim: cfg.headDim });

  let x = context; // [tokens · 12, 2560]
  for (const b of w.layerwise) x = fusionBlock(graph, b, cfg, x, tokens * TEXT_LAYERS, acrossLayers);
  const merged = graph.buffer(tokens * cfg.hidden * 4);
  graph.run(layerSumWgsl, [graph.buffer(new Uint32Array([tokens, TEXT_LAYERS, cfg.hidden])), x, w.projector, merged], layerSumWorkgroups(tokens, cfg.hidden));
  x = merged;
  for (const b of w.refiner) x = fusionBlock(graph, b, cfg, x, tokens, acrossTokens);
  const fused = x;

  const h = norm(graph, fused, w.txtNorm, tokens, cfg.hidden, noMod(graph, cfg.hidden));
  const out = linear(graph, gelu(graph, linear(graph, h, w.txt1, tokens), tokens * w.txt1.n), w.txt3, tokens);
  return { fused, out };
}

/**
 * `context`: text hidden states [tokens, 12 · 2560]. `textCache`: the text part from an earlier step of
 * the same image (it doesn't depend on the timestep), so it runs once per image.
 */
export type Inputs = {
  latent: Float32Array;
  height: number;
  width: number;
  timestep: number;
  context: Float32Array;
  textTokens: number;
  textCache?: GPUBuffer;
};

/**
 * One denoising step: predicted velocity for the latent, as [16, H, W].
 * `steps` collects the output of each part (for comparing with a reference); `onBlock` reports progress.
 */
export async function transformer(graph: Graph, w: TransformerWeights, input: Inputs, onBlock?: (i: number) => void, cfg = KREA_BLOCK) {
  const { height, width, textTokens: L } = input;
  const { rows, cols } = patchGrid(height, width);
  const image = rows * cols;
  const tokens = L + image;
  const hidden = cfg.hidden;
  const steps: Record<string, GPUBuffer> = {};

  const { t, tvec } = time(graph, w, input.timestep);
  const txt = input.textCache ? { out: input.textCache } : text(graph, w, graph.buffer(input.context), L);
  const img = linear(graph, graph.buffer(patchify(input.latent, height, width)), w.first, image);
  Object.assign(steps, { tmlp: t, tproj: tvec, txtmlp: txt.out, first: img }, "fused" in txt ? { txtfusion: txt.fused } : {});

  let x = graph.buffer(tokens * hidden * 4);
  graph.copy(txt.out, 0, x, 0, L * hidden * 4);
  graph.copy(img, 0, x, L * hidden * 4, image * hidden * 4);

  const tables = ropeTables(positions(L, rows, cols));
  const rope = { cos: graph.buffer(tables.cos), sin: graph.buffer(tables.sin), pairs: tables.pairs, tokens };

  await runAhead(
    w.blocks.length,
    (i) => {
      const mark = graph.mark();
      const next = block(graph, w.blocks[i], cfg, x, tvec, rope);
      graph.release(mark, [next]); // free this block's temporaries (submits the block)
      graph.free(x); // the previous block's input/output is no longer needed
      x = next;
    },
    () => graph.device.queue.onSubmittedWorkDone(), // the GPU never waits for the CPU; memory stays bounded
    (i) => onBlock?.(i),
  );
  steps.blocks = x;

  const scale = add(graph, t, w.lastScale, hidden);
  const shift = add(graph, t, w.lastShift, hidden);
  const last = linear(graph, norm(graph, x, w.lastNorm, tokens, hidden, { scale, shift }), w.lastLinear, tokens);
  steps.last = last;

  const all = new Float32Array(await graph.read(last, tokens * 64 * 4));
  return { velocity: unpatchify(all.subarray(L * 64), height, width), steps };
}
