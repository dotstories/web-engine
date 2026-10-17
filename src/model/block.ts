import { Graph } from "../gpu/graph";
import { attention } from "../kernels/attention";
import { add, gatedAdd, linear, noMod, norm, rotate, sigmoidGate, swiglu, type Linear, type Rope } from "./ops";

/** Sizes of a Krea 2 transformer block (tests use a tiny one). hidden must be a multiple of 64. */
export type BlockConfig = { hidden: number; mlp: number; heads: number; kvHeads: number; headDim: number; eps: number };

export const KREA_BLOCK: BlockConfig = { hidden: 6144, mlp: 16384, heads: 48, kvHeads: 12, headDim: 128, eps: 1e-5 };

/** The attention + MLP weights every Krea 2 block has (the text fusion blocks too). */
export type AttentionMlpWeights = {
  wq: Linear;
  wk: Linear;
  wv: Linear;
  gate: Linear;
  wo: Linear;
  mlpGate: Linear;
  mlpUp: Linear;
  mlpDown: Linear;
  norm1: GPUBuffer;
  norm2: GPUBuffer;
  normQ: GPUBuffer;
  normK: GPUBuffer;
};

/** A single-stream block also has its own modulation table (6 × hidden), added to the timestep vector. */
export type BlockWeights = AttentionMlpWeights & { table: GPUBuffer };

/** The 6 modulation vectors, in ComfyUI/diffusers order. */
export const MOD = ["prescale", "preshift", "pregate", "postscale", "postshift", "postgate"] as const;

/** mod = shared timestep vector tvec (6 × hidden) + the block's table; returns the 6 parts as bindings. */
export function modulation(graph: Graph, tvec: GPUBuffer, table: GPUBuffer, hidden: number) {
  const mod = add(graph, tvec, table, 6 * hidden);
  const bytes = hidden * 4;
  return Object.fromEntries(MOD.map((name, i) => [name, Graph.slice(mod, i * bytes, bytes)])) as Record<(typeof MOD)[number], ReturnType<typeof Graph.slice>>;
}

/** Attention part shared by the blocks: norm(q), norm(k) per head, optional RoPE, attend, gate, out. */
function attend(
  graph: Graph,
  w: AttentionMlpWeights,
  cfg: BlockConfig,
  h: GPUBuffer,
  rows: number,
  attn: (q: GPUBuffer, k: GPUBuffer, v: GPUBuffer) => GPUBuffer,
  rope?: Rope,
): GPUBuffer {
  const { hidden, heads, kvHeads, headDim, eps } = cfg;
  const none = noMod(graph, headDim); // q/k norm has no modulation
  // Each temporary is freed right after its last use, so the next one reuses its memory: at 2 MP a block's
  // temporaries would otherwise hold ~5.4 GB next to 13 GB of weights, past what fits on a 16 GB GPU.
  const projected = (w: Linear, n: number, normW: GPUBuffer) => used(graph, linear(graph, h, w, rows), (y) => norm(graph, y, normW, rows * n, headDim, none, eps));
  let q = projected(w.wq, heads, w.normQ);
  let k = projected(w.wk, kvHeads, w.normK);
  if (rope) [q, k] = [used(graph, q, (y) => rotate(graph, y, heads, rope)), used(graph, k, (y) => rotate(graph, y, kvHeads, rope))];
  const v = linear(graph, h, w.wv, rows);
  const o = attn(q, k, v);
  [q, k, v].forEach((b) => graph.free(b));
  const g = linear(graph, h, w.gate, rows);
  const og = sigmoidGate(graph, o, g, rows * hidden);
  graph.free(o);
  graph.free(g);
  return used(graph, og, (y) => linear(graph, y, w.wo, rows));
}

/** f(x), then x is freed (its last use is queued). */
function used(graph: Graph, x: GPUBuffer, f: (x: GPUBuffer) => GPUBuffer): GPUBuffer {
  const y = f(x);
  graph.free(x);
  return y;
}

function mlp(graph: Graph, w: AttentionMlpWeights, cfg: BlockConfig, h: GPUBuffer, rows: number): GPUBuffer {
  const gate = linear(graph, h, w.mlpGate, rows);
  const up = linear(graph, h, w.mlpUp, rows);
  const s = swiglu(graph, gate, up, rows * cfg.mlp);
  graph.free(gate);
  graph.free(up);
  return used(graph, s, (y) => linear(graph, y, w.mlpDown, rows));
}

/**
 * One Krea 2 single-stream block (ComfyUI SingleStreamBlock / diffusers Krea2TransformerBlock):
 *   h = (1 + prescale) · norm1(x) + preshift
 *   x = x + pregate · wo(attention(rope(norm_q(wq h)), rope(norm_k(wk h)), wv h) · sigmoid(gate h))
 *   h = (1 + postscale) · norm2(x) + postshift
 *   x = x + postgate · down(silu(mlp_gate h) · up h)
 * x: [tokens, hidden] → new [tokens, hidden]. tvec: the shared timestep vector [6 × hidden].
 */
export function block(graph: Graph, w: BlockWeights, cfg: BlockConfig, x: GPUBuffer, tvec: GPUBuffer, rope: Rope): GPUBuffer {
  const { hidden, heads, kvHeads, headDim, eps } = cfg;
  const t = rope.tokens;
  const mod = modulation(graph, tvec, w.table, hidden);
  const full = (q: GPUBuffer, k: GPUBuffer, v: GPUBuffer) => attention(graph, q, k, v, { tokens: t, heads, kvHeads, dim: headDim });

  const h = norm(graph, x, w.norm1, t, hidden, { scale: mod.prescale, shift: mod.preshift }, eps);
  const a = used(graph, h, (y) => attend(graph, w, cfg, y, t, full, rope));
  const x1 = used(graph, a, (y) => gatedAdd(graph, x, y, mod.pregate, t * hidden, hidden));
  const h2 = norm(graph, x1, w.norm2, t, hidden, { scale: mod.postscale, shift: mod.postshift }, eps);
  const m = used(graph, h2, (y) => mlp(graph, w, cfg, y, t));
  const out = gatedAdd(graph, x1, m, mod.postgate, t * hidden, hidden);
  [m, x1].forEach((b) => graph.free(b));
  return out;
}

/**
 * A text fusion block (ComfyUI TextFusionBlock): plain pre-norm residual, no modulation, no RoPE.
 *   x = x + attn(norm1(x));  x = x + mlp(norm2(x))
 * `attn` decides what attends to what (each token's 12 layers, or all tokens).
 */
export function fusionBlock(
  graph: Graph,
  w: AttentionMlpWeights,
  cfg: BlockConfig,
  x: GPUBuffer,
  rows: number,
  attn: (q: GPUBuffer, k: GPUBuffer, v: GPUBuffer) => GPUBuffer,
): GPUBuffer {
  const { hidden, eps } = cfg;
  const none = noMod(graph, hidden);
  const a = used(graph, norm(graph, x, w.norm1, rows, hidden, none, eps), (h) => attend(graph, w, cfg, h, rows, attn));
  const x1 = used(graph, a, (y) => add(graph, x, y, rows * hidden));
  const m = used(graph, norm(graph, x1, w.norm2, rows, hidden, none, eps), (h) => mlp(graph, w, cfg, h, rows));
  const out = add(graph, x1, m, rows * hidden);
  [m, x1].forEach((b) => graph.free(b));
  return out;
}
