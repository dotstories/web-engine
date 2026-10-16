import type { Graph } from "../gpu/graph";
import { attention } from "../kernels/attention";
import { ropeTables } from "../kernels/rope";
import { templateEnd } from "../text/prompt";
import { toFloat32 } from "../weights/fp8";
import type { Safetensors, TensorInfo } from "../weights/safetensors";
import { add, linear, noMod, norm, rotate, swiglu, type Linear, type Rope } from "./ops";
import { loadFloats, loadLinear } from "./weights";

/**
 * Krea 2's text encoder: the text part of Qwen3-VL 4B (ComfyUI comfy/text_encoders/krea2.py + llama.py).
 * A Qwen3 decoder: plain RMSNorms (eps 1e-6), q/k norm per head, split-half RoPE (θ = 5·10⁶; for text the
 * 3 MRoPE axes share one position), causal grouped-query attention (32 q / 8 kv heads), SwiGLU MLP.
 * Krea 2 taps the hidden states after layers 2, 5, …, 35 (no final norm), so only 35 of the 36 layers run.
 */
export const QWEN3VL_4B = { hidden: 2560, mlp: 9728, heads: 32, kvHeads: 8, headDim: 128, eps: 1e-6, theta: 5e6 };
export const TAPS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35];
export const LAYERS = TAPS.at(-1)!;

export type DecoderLayer = { q: Linear; k: Linear; v: Linear; o: Linear; gate: Linear; up: Linear; down: Linear; ln1: GPUBuffer; ln2: GPUBuffer; qNorm: GPUBuffer; kNorm: GPUBuffer };
export type TextEncoderWeights = { layers: DecoderLayer[]; file: Blob; embeddings: TensorInfo };

export async function loadTextEncoder(graph: Graph, file: Blob, st: Safetensors, onProgress?: (done: number, total: number) => void): Promise<TextEncoderWeights> {
  const embeddings = st.tensors.get("model.embed_tokens.weight");
  if (!embeddings || embeddings.dtype !== "BF16") throw new Error("Missing model.embed_tokens.weight (BF16): is this the Qwen3-VL 4B text encoder?");
  const layers: DecoderLayer[] = [];
  for (let i = 0; i < LAYERS; i++) {
    const p = `model.layers.${i}`;
    const lin = (name: string) => loadLinear(graph, file, st, `${p}.${name}`);
    const vec = async (name: string) => graph.buffer(await loadFloats(file, st, `${p}.${name}`));
    layers.push({
      q: await lin("self_attn.q_proj"),
      k: await lin("self_attn.k_proj"),
      v: await lin("self_attn.v_proj"),
      o: await lin("self_attn.o_proj"),
      gate: await lin("mlp.gate_proj"),
      up: await lin("mlp.up_proj"),
      down: await lin("mlp.down_proj"),
      ln1: await vec("input_layernorm.weight"),
      ln2: await vec("post_attention_layernorm.weight"),
      qNorm: await vec("self_attn.q_norm.weight"),
      kNorm: await vec("self_attn.k_norm.weight"),
    });
    onProgress?.(i + 1, LAYERS);
    await graph.device.queue.onSubmittedWorkDone(); // bounded RAM while uploading
  }
  return { layers, file, embeddings };
}

/** Only the rows of the prompt's tokens, read straight from the file (the full table is 778 MB). */
export async function embed(file: Blob, table: TensorInfo, ids: number[]): Promise<Float32Array<ArrayBuffer>> {
  const [vocab, hidden] = table.shape;
  const rowBytes = hidden * 2; // bf16
  const out = new Float32Array(ids.length * hidden);
  for (const [t, id] of ids.entries()) {
    if (id < 0 || id >= vocab) throw new Error(`Token id ${id} is outside the vocabulary`);
    const start = table.offset + id * rowBytes;
    out.set(toFloat32("BF16", new Uint8Array(await file.slice(start, start + rowBytes).arrayBuffer())), t * hidden);
  }
  return out;
}

function decoderLayer(graph: Graph, w: DecoderLayer, x: GPUBuffer, tokens: number, rope: Rope): GPUBuffer {
  const { hidden, mlp, heads, kvHeads, headDim, eps } = QWEN3VL_4B;
  const plain = (b: GPUBuffer, weight: GPUBuffer, rows: number, d: number) => norm(graph, b, weight, rows, d, noMod(graph, d), eps, 0);

  const h = plain(x, w.ln1, tokens, hidden);
  const q = rotate(graph, plain(linear(graph, h, w.q, tokens), w.qNorm, tokens * heads, headDim), heads, rope, "half");
  const k = rotate(graph, plain(linear(graph, h, w.k, tokens), w.kNorm, tokens * kvHeads, headDim), kvHeads, rope, "half");
  const o = attention(graph, q, k, linear(graph, h, w.v, tokens), { tokens, heads, kvHeads, dim: headDim, causal: true });
  const x1 = add(graph, x, linear(graph, o, w.o, tokens), tokens * hidden);

  const h2 = plain(x1, w.ln2, tokens, hidden);
  const m = swiglu(graph, linear(graph, h2, w.gate, tokens), linear(graph, h2, w.up, tokens), tokens * mlp);
  return add(graph, x1, linear(graph, m, w.down, tokens), tokens * hidden);
}

/**
 * Token ids (the whole template) → Krea 2's conditioning for the tokens after "<|im_start|>user\n":
 * [kept tokens, 12 · 2560], each token's 12 tapped layers side by side (ComfyUI's layout).
 */
export async function encode(graph: Graph, w: TextEncoderWeights, ids: number[], onLayer?: (i: number) => void) {
  const { hidden, headDim, theta } = QWEN3VL_4B;
  const tokens = ids.length;
  const tables = ropeTables(ids.map((_, t) => [t]), [headDim], theta);
  const rope = { cos: graph.buffer(tables.cos), sin: graph.buffer(tables.sin), pairs: tables.pairs, tokens };

  let x = graph.buffer(await embed(w.file, w.embeddings, ids));
  const taps = new Map<number, GPUBuffer>();
  for (let i = 0; i < LAYERS; i++) {
    if (TAPS.includes(i)) taps.set(i, x); // hidden_states[i]: the input of layer i
    const mark = graph.mark();
    const next = decoderLayer(graph, w.layers[i], x, tokens, rope);
    graph.release(mark, [next]);
    if (!TAPS.includes(i)) graph.free(x);
    x = next;
    onLayer?.(i);
    await graph.device.queue.onSubmittedWorkDone();
  }
  taps.set(LAYERS, x);

  const start = templateEnd(ids);
  const kept = tokens - start;
  const cond = new Float32Array(kept * TAPS.length * hidden);
  for (const [l, layer] of TAPS.entries()) {
    const states = new Float32Array(await graph.read(taps.get(layer)!, tokens * hidden * 4));
    for (let t = 0; t < kept; t++) cond.set(states.subarray((start + t) * hidden, (start + t + 1) * hidden), (t * TAPS.length + l) * hidden);
  }
  return { cond, kept };
}
