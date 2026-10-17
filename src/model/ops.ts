import type { Binding, Graph } from "../gpu/graph";
import {
  addBiasWgsl,
  addWgsl,
  elementwiseWorkgroups,
  gatedAddWgsl,
  geluTanhWgsl,
  sigmoidGateWgsl,
  siluWgsl,
  swigluWgsl,
} from "../kernels/elementwise";
import { matmulBf16Wgsl, matmulF32Wgsl, matmulFp8Wgsl, matmulParams, matmulWgsl, matmulWorkgroups, TILE, type MatmulMode } from "../kernels/matmul";
import { layerSumWgsl, layerSumWorkgroups } from "../kernels/text";
import { rmsnormParams, rmsnormWgsl, rmsnormWorkgroups } from "../kernels/rmsnorm";
import { ropeKernel, ropeWorkgroups, type Pairing } from "../kernels/rope";

/** The kernels as graph steps: each takes GPU buffers, queues one dispatch and returns its output buffer. */

/**
 * A linear layer: W [n, k] as stored (fp8 packed 4 per u32 with a scalar weight_scale, bf16 packed
 * 2 per u32, or f32), an optional f32 bias [n], and an optional LoRA applied at run time:
 * down [rank, k] and up [n, rank] with up.scale = strength · alpha / rank (rank a multiple of 4).
 */
export type Linear = {
  buffer: GPUBuffer;
  dtype: "fp8" | "bf16" | "f32";
  scale: number;
  n: number;
  k: number;
  bias?: GPUBuffer;
  lora?: { down: Linear; up: Linear };
};

const MATMUL = { fp8: matmulFp8Wgsl, bf16: matmulBf16Wgsl, f32: matmulF32Wgsl };
const variants = (mode: MatmulMode) => ({ fp8: matmulWgsl("fp8", "rows", mode), bf16: matmulWgsl("bf16", "rows", mode), f32: matmulWgsl("f32", "rows", mode) });
const SPLIT_K = variants({ splitK: true });
const ACCUMULATE = variants({ accumulate: true });
const BUSY = 256; // workgroups that keep a big GPU busy
const MIN_SLICE = 256; // K values per slice, at least

/** How many K slices make x[m, k] · Wᵀ (n narrow) fill the GPU: 1 = no split. */
export function kSplitFor(m: number, n: number, k: number): number {
  const tiles = Math.ceil(m / TILE) * Math.ceil(n / TILE);
  return Math.max(1, Math.min(Math.ceil(BUSY / tiles), Math.floor(k / MIN_SLICE)));
}

/** y = x · Wᵀ like linear(), split over K when the output is too narrow to fill the GPU (a LoRA's down). */
function narrowLinear(graph: Graph, x: GPUBuffer, w: Linear, m: number): GPUBuffer {
  const split = kSplitFor(m, w.n, w.k);
  if (split === 1) return linear(graph, x, w, m);
  const shape = { m, n: w.n, k: w.k };
  const partials = graph.buffer(split * m * w.n * 4);
  graph.run(SPLIT_K[w.dtype], [graph.buffer(matmulParams(shape, w.scale, { kSplit: split })), x, w.buffer, partials], [...matmulWorkgroups(shape), split]);
  const y = graph.buffer(m * w.n * 4);
  const params = graph.buffer(new Uint32Array([1, split, m * w.n]));
  graph.run(layerSumWgsl, [params, partials, graph.buffer(new Float32Array(split).fill(1)), y], layerSumWorkgroups(1, m * w.n));
  return y;
}

/** y[m, n] = x[m, k] · Wᵀ (+ bias) (+ up.scale · (x · downᵀ) · upᵀ, added in place). Same as ComfyUI's merged W + Δ, without touching W. */
export function linear(graph: Graph, x: GPUBuffer, w: Linear, m: number): GPUBuffer {
  const shape = { m, n: w.n, k: w.k };
  const y = graph.buffer(m * w.n * 4);
  graph.run(MATMUL[w.dtype], [graph.buffer(matmulParams(shape, w.scale)), x, w.buffer, y], matmulWorkgroups(shape));
  const out = w.bias ? elementwise(graph, addBiasWgsl, [y, w.bias], m * w.n, w.n) : y;
  if (!w.lora) return out;
  const { down, up } = w.lora;
  if (down.k !== w.k || up.n !== w.n || up.k !== down.n || down.n % 4) throw new Error("LoRA shapes do not fit the linear layer");
  const t = narrowLinear(graph, x, down, m);
  const upShape = { m, n: up.n, k: up.k };
  graph.run(ACCUMULATE[up.dtype], [graph.buffer(matmulParams(upShape, up.scale)), t, up.buffer, out], matmulWorkgroups(upShape)); // out += …
  return out;
}

export type Mod = { scale: Binding; shift: Binding };

/** RMSNorm per row of length d, then (1 + scale) · … + shift. offset 1: zero-centered weight (Krea 2), 0: plain. */
export function norm(graph: Graph, x: GPUBuffer, weight: GPUBuffer, rows: number, d: number, mod: Mod, eps = 1e-5, offset = 1): GPUBuffer {
  const out = graph.buffer(rows * d * 4);
  graph.run(rmsnormWgsl, [graph.buffer(rmsnormParams(rows, d, eps, offset)), x, weight, mod.scale, mod.shift, out], rmsnormWorkgroups(rows));
  return out;
}

/** No modulation: zero scale and shift of length d. */
export const noMod = (graph: Graph, d: number): Mod => ({ scale: graph.buffer(d * 4), shift: graph.buffer(d * 4) });

export type Rope = { cos: GPUBuffer; sin: GPUBuffer; pairs: number; tokens: number };

export function rotate(graph: Graph, x: GPUBuffer, heads: number, rope: Rope, pairing: Pairing = "interleaved"): GPUBuffer {
  const out = graph.buffer(rope.tokens * heads * rope.pairs * 2 * 4);
  graph.run(ropeKernel(pairing), [graph.buffer(new Uint32Array([rope.tokens, heads, rope.pairs])), x, rope.cos, rope.sin, out], ropeWorkgroups(rope.tokens, rope.pairs));
  return out;
}

function elementwise(graph: Graph, code: string, inputs: Binding[], n: number, d = 1): GPUBuffer {
  const out = graph.buffer(n * 4);
  graph.run(code, [graph.buffer(new Uint32Array([n, d])), ...inputs, out], elementwiseWorkgroups(n));
  return out;
}

/** silu(gate) · up */
export const swiglu = (graph: Graph, gate: GPUBuffer, up: GPUBuffer, n: number) => elementwise(graph, swigluWgsl, [gate, up], n);
/** a · sigmoid(gate) */
export const sigmoidGate = (graph: Graph, a: GPUBuffer, gate: GPUBuffer, n: number) => elementwise(graph, sigmoidGateWgsl, [a, gate], n);
/** x + g[column] · h, with g of length d */
export const gatedAdd = (graph: Graph, x: GPUBuffer, h: GPUBuffer, g: Binding, n: number, d: number) =>
  elementwise(graph, gatedAddWgsl, [x, h, g], n, d);
/** x + h */
export const add = (graph: Graph, x: Binding, h: Binding, n: number) => elementwise(graph, addWgsl, [x, h], n);
/** GELU, tanh form */
export const gelu = (graph: Graph, x: GPUBuffer, n: number) => elementwise(graph, geluTanhWgsl, [x], n);
/** x · sigmoid(x) */
export const silu = (graph: Graph, x: GPUBuffer, n: number) => elementwise(graph, siluWgsl, [x], n);
