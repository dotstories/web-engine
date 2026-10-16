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
import { matmulBf16Wgsl, matmulFp8Wgsl, matmulParams, matmulWorkgroups } from "../kernels/matmul";
import { rmsnormParams, rmsnormWgsl, rmsnormWorkgroups } from "../kernels/rmsnorm";
import { ropeWgsl, ropeWorkgroups } from "../kernels/rope";

/** The kernels as graph steps: each takes GPU buffers, queues one dispatch and returns its output buffer. */

/**
 * A linear layer from the model file: W [n, k] as stored (fp8 packed 4 per u32 with a scalar
 * weight_scale, or bf16 packed 2 per u32), and an optional f32 bias [n].
 */
export type Linear = { buffer: GPUBuffer; dtype: "fp8" | "bf16"; scale: number; n: number; k: number; bias?: GPUBuffer };

/** y[m, n] = x[m, k] · Wᵀ (+ bias) */
export function linear(graph: Graph, x: GPUBuffer, w: Linear, m: number): GPUBuffer {
  const shape = { m, n: w.n, k: w.k };
  const y = graph.buffer(m * w.n * 4);
  const code = w.dtype === "fp8" ? matmulFp8Wgsl : matmulBf16Wgsl;
  graph.run(code, [graph.buffer(matmulParams(shape, w.scale)), x, w.buffer, y], matmulWorkgroups(shape));
  return w.bias ? elementwise(graph, addBiasWgsl, [y, w.bias], m * w.n, w.n) : y;
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

export function rotate(graph: Graph, x: GPUBuffer, heads: number, rope: Rope): GPUBuffer {
  const out = graph.buffer(rope.tokens * heads * rope.pairs * 2 * 4);
  graph.run(ropeWgsl, [graph.buffer(new Uint32Array([rope.tokens, heads, rope.pairs])), x, rope.cos, rope.sin, out], ropeWorkgroups(rope.tokens, rope.pairs));
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
