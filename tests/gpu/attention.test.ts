import { describe, expect, it } from "vitest";
import { getDevice } from "../../src/gpu/compute";
import { Graph } from "../../src/gpu/graph";
import { attention, type AttentionShape } from "../../src/kernels/attention";

let seed = 1;
const random = (n: number, size = 2) => Float32Array.from({ length: n }, () => (((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32) * 2 - 1) * size);

/** float64 reference: softmax(q·kᵀ/√d)·v per head, kv head = ⌊h / group⌋. */
function reference(q: Float32Array, k: Float32Array, v: Float32Array, { tokens, heads, kvHeads, dim }: AttentionShape) {
  const out = new Float64Array(tokens * heads * dim);
  const group = heads / kvHeads;
  for (let h = 0; h < heads; h++) {
    const g = Math.floor(h / group);
    for (let i = 0; i < tokens; i++) {
      const s = Array.from({ length: tokens }, (_, j) => {
        let dot = 0;
        for (let d = 0; d < dim; d++) dot += q[(i * heads + h) * dim + d] * k[(j * kvHeads + g) * dim + d];
        return dot / Math.sqrt(dim);
      });
      const m = Math.max(...s);
      const e = s.map((x) => Math.exp(x - m));
      const sum = e.reduce((a, b) => a + b, 0);
      for (let d = 0; d < dim; d++) {
        let o = 0;
        for (let j = 0; j < tokens; j++) o += (e[j] / sum) * v[(j * kvHeads + g) * dim + d];
        out[(i * heads + h) * dim + d] = o;
      }
    }
  }
  return out;
}

describe("attention", () => {
  it.each([
    { tokens: 37, heads: 4, kvHeads: 1, dim: 128 }, // odd token count, all heads share one kv head
    { tokens: 64, heads: 8, kvHeads: 2, dim: 128 }, // Krea 2's 4 query heads per kv head
  ])("matches the CPU reference: $tokens tokens, $heads heads, $kvHeads kv heads", async (shape) => {
    const { tokens, heads, kvHeads, dim } = shape;
    const [q, k, v] = [random(tokens * heads * dim), random(tokens * kvHeads * dim), random(tokens * kvHeads * dim)];
    const graph = new Graph(await getDevice());
    try {
      const out = attention(graph, graph.buffer(q), graph.buffer(k), graph.buffer(v), shape);
      const gpu = new Float32Array(await graph.read(out, tokens * heads * dim * 4));
      const cpu = reference(q, k, v, shape);
      expect(gpu.reduce((m, x, i) => Math.max(m, Math.abs(x - cpu[i])), 0)).toBeLessThan(1e-5);
    } finally {
      graph.destroy();
    }
  });

  it("refuses heads that don't split evenly over kv heads", async () => {
    const graph = new Graph(await getDevice());
    const b = graph.buffer(4);
    expect(() => attention(graph, b, b, b, { tokens: 1, heads: 3, kvHeads: 2, dim: 4 })).toThrow(/multiple/);
    graph.destroy();
  });
});
