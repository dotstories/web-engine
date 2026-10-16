import type { Graph } from "../gpu/graph";
import { attention } from "../kernels/attention";
import { gatedAdd, linear, norm, rotate, sigmoidGate, swiglu, type Linear, type Rope } from "./ops";

/** Sizes of a Krea 2 transformer block (tests use a tiny one). */
export type BlockConfig = { hidden: number; mlp: number; heads: number; kvHeads: number; headDim: number; eps: number };

export const KREA_BLOCK: BlockConfig = { hidden: 6144, mlp: 16384, heads: 48, kvHeads: 12, headDim: 128, eps: 1e-5 };

/** One block's weights on the GPU. `table` (6 × hidden) stays on the CPU: it is added to the timestep modulation. */
export type BlockWeights = {
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
  table: Float32Array;
};

/** The 6 modulation vectors, in ComfyUI/diffusers order. */
export const MOD = ["prescale", "preshift", "pregate", "postscale", "postshift", "postgate"] as const;

/** mod = shared timestep modulation (6 × hidden) + the block's own table. */
export function modulation(graph: Graph, temb: Float32Array, table: Float32Array, hidden: number) {
  const vectors = MOD.map((_, i) => {
    const v = new Float32Array(hidden);
    for (let j = 0; j < hidden; j++) v[j] = temb[i * hidden + j] + table[i * hidden + j];
    return graph.buffer(v);
  });
  return Object.fromEntries(MOD.map((name, i) => [name, vectors[i]])) as Record<(typeof MOD)[number], GPUBuffer>;
}

/**
 * One Krea 2 single-stream block (ComfyUI SingleStreamBlock / diffusers Krea2TransformerBlock):
 *   h = (1 + prescale) · norm1(x) + preshift
 *   x = x + pregate · wo(attention(rope(norm_q(wq h)), rope(norm_k(wk h)), wv h) · sigmoid(gate h))
 *   h = (1 + postscale) · norm2(x) + postshift
 *   x = x + postgate · down(silu(mlp_gate h) · up h)
 * x: [tokens, hidden] → new [tokens, hidden].
 */
export function block(graph: Graph, w: BlockWeights, cfg: BlockConfig, x: GPUBuffer, temb: Float32Array, rope: Rope): GPUBuffer {
  const { hidden, mlp, heads, kvHeads, headDim, eps } = cfg;
  const t = rope.tokens;
  const mod = modulation(graph, temb, w.table, hidden);
  const none = { scale: graph.buffer(headDim * 4), shift: graph.buffer(headDim * 4) }; // q/k norm has no modulation

  const h = norm(graph, x, w.norm1, t, hidden, { scale: mod.prescale, shift: mod.preshift }, eps);
  const q = rotate(graph, norm(graph, linear(graph, h, w.wq, t), w.normQ, t * heads, headDim, none, eps), heads, rope);
  const k = rotate(graph, norm(graph, linear(graph, h, w.wk, t), w.normK, t * kvHeads, headDim, none, eps), kvHeads, rope);
  const v = linear(graph, h, w.wv, t);
  const o = attention(graph, q, k, v, { tokens: t, heads, kvHeads, dim: headDim });
  const a = linear(graph, sigmoidGate(graph, o, linear(graph, h, w.gate, t), t * hidden), w.wo, t);
  const x1 = gatedAdd(graph, x, a, mod.pregate, t * hidden, hidden);

  const h2 = norm(graph, x1, w.norm2, t, hidden, { scale: mod.postscale, shift: mod.postshift }, eps);
  const m = swiglu(graph, linear(graph, h2, w.mlpGate, t), linear(graph, h2, w.mlpUp, t), t * mlp);
  return gatedAdd(graph, x1, linear(graph, m, w.mlpDown, t), mod.postgate, t * hidden, hidden);
}
