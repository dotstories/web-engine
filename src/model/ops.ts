import type { Graph } from "../gpu/graph";
import { elementwiseWorkgroups, gatedAddWgsl, sigmoidGateWgsl, swigluWgsl } from "../kernels/elementwise";
import { matmulFp8Wgsl, matmulParams, matmulWorkgroups } from "../kernels/matmul";
import { rmsnormParams, rmsnormWgsl, rmsnormWorkgroups } from "../kernels/rmsnorm";
import { ropeWgsl, ropeWorkgroups } from "../kernels/rope";

/** The kernels as graph steps: each takes GPU buffers, queues one dispatch and returns its output buffer. */

/** An fp8 linear layer from the model file: W [n, k] packed 4 per u32, real value = fp8 × scale. */
export type Linear = { buffer: GPUBuffer; scale: number; n: number; k: number };

/** y[m, n] = x[m, k] · Wᵀ */
export function linear(graph: Graph, x: GPUBuffer, w: Linear, m: number): GPUBuffer {
  const shape = { m, n: w.n, k: w.k };
  const y = graph.buffer(m * w.n * 4);
  graph.run(matmulFp8Wgsl, [graph.buffer(matmulParams(shape, w.scale)), x, w.buffer, y], matmulWorkgroups(shape));
  return y;
}

/** Zero-centered RMSNorm per row of length d, then (1 + scale) · … + shift. */
export function norm(
  graph: Graph,
  x: GPUBuffer,
  weight: GPUBuffer,
  rows: number,
  d: number,
  mod: { scale: GPUBuffer; shift: GPUBuffer },
  eps = 1e-5,
): GPUBuffer {
  const out = graph.buffer(rows * d * 4);
  graph.run(rmsnormWgsl, [graph.buffer(rmsnormParams(rows, d, eps, 1)), x, weight, mod.scale, mod.shift, out], rmsnormWorkgroups(rows));
  return out;
}

export type Rope = { cos: GPUBuffer; sin: GPUBuffer; pairs: number; tokens: number };

export function rotate(graph: Graph, x: GPUBuffer, heads: number, rope: Rope): GPUBuffer {
  const out = graph.buffer(rope.tokens * heads * rope.pairs * 2 * 4);
  graph.run(ropeWgsl, [graph.buffer(new Uint32Array([rope.tokens, heads, rope.pairs])), x, rope.cos, rope.sin, out], ropeWorkgroups(rope.tokens, rope.pairs));
  return out;
}

function elementwise(graph: Graph, code: string, inputs: GPUBuffer[], n: number, d = 1): GPUBuffer {
  const out = graph.buffer(n * 4);
  graph.run(code, [graph.buffer(new Uint32Array([n, d])), ...inputs, out], elementwiseWorkgroups(n));
  return out;
}

/** silu(gate) · up */
export const swiglu = (graph: Graph, gate: GPUBuffer, up: GPUBuffer, n: number) => elementwise(graph, swigluWgsl, [gate, up], n);
/** a · sigmoid(gate) */
export const sigmoidGate = (graph: Graph, a: GPUBuffer, gate: GPUBuffer, n: number) => elementwise(graph, sigmoidGateWgsl, [a, gate], n);
/** x + g[column] · h, with g of length d */
export const gatedAdd = (graph: Graph, x: GPUBuffer, h: GPUBuffer, g: GPUBuffer, n: number, d: number) =>
  elementwise(graph, gatedAddWgsl, [x, h, g], n, d);
