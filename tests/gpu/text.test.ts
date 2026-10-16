import { describe, expect, it } from "vitest";
import { getDevice } from "../../src/gpu/compute";
import { add, addBias, geluTanh } from "../../src/kernels/elementwise";
import { layerSum, shortAttention } from "../../src/kernels/text";

let seed = 3;
const random = (n: number, size = 2) => Float32Array.from({ length: n }, () => (((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32) * 2 - 1) * size);
const maxError = (a: Float32Array, ref: (i: number) => number) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - ref(i))), 0);

describe("shortAttention", () => {
  it("attends inside each short sequence, like Krea 2's 12 text layers per token", async () => {
    const [batch, seq, heads, dim] = [5, 12, 3, 128];
    const [q, k, v] = [random(batch * seq * heads * dim), random(batch * seq * heads * dim), random(batch * seq * heads * dim)];
    const gpu = await shortAttention(await getDevice(), q, k, v, { batch, seq, heads, dim });
    const at = (b: number, i: number, h: number, d: number) => ((b * seq + i) * heads + h) * dim + d;
    let err = 0;
    for (let b = 0; b < batch; b++)
      for (let h = 0; h < heads; h++)
        for (let i = 0; i < seq; i++) {
          const s = Array.from({ length: seq }, (_, j) => {
            let dot = 0;
            for (let d = 0; d < dim; d++) dot += q[at(b, i, h, d)] * k[at(b, j, h, d)];
            return dot / Math.sqrt(dim);
          });
          const m = Math.max(...s);
          const e = s.map((x) => Math.exp(x - m));
          const sum = e.reduce((a, c) => a + c, 0);
          for (let d = 0; d < dim; d++) {
            let o = 0;
            for (let j = 0; j < seq; j++) o += (e[j] / sum) * v[at(b, j, h, d)];
            err = Math.max(err, Math.abs(gpu[at(b, i, h, d)] - o));
          }
        }
    expect(err).toBeLessThan(1e-5);
  });
});

describe("layerSum", () => {
  it("merges each token's layers with one weight per layer", async () => {
    const [tokens, layers, dim] = [7, 12, 40];
    const [x, w] = [random(tokens * layers * dim), random(layers)];
    const gpu = await layerSum(await getDevice(), x, w, tokens, dim);
    expect(maxError(gpu, (i) => {
      const [t, d] = [Math.floor(i / dim), i % dim];
      return w.reduce((s, wl, l) => s + wl * x[(t * layers + l) * dim + d], 0);
    })).toBeLessThan(1e-5);
  });
});

describe("text MLP pieces", () => {
  it("addBias adds one value per column", async () => {
    const [x, b] = [random(60), random(6)];
    expect(maxError(await addBias(await getDevice(), x, b), (i) => x[i] + b[i % 6])).toBeLessThan(1e-6);
  });

  it("geluTanh matches PyTorch's tanh GELU, also far from zero", async () => {
    const x = Float32Array.from([...random(200, 8), -60, 60, 0]);
    const gelu = (v: number) => 0.5 * v * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (v + 0.044715 * v ** 3)));
    expect(maxError(await geluTanh(await getDevice(), x), (i) => gelu(x[i]))).toBeLessThan(1e-5);
  });

  it("add sums two tensors", async () => {
    const [x, h] = [random(50), random(50)];
    expect(maxError(await add(await getDevice(), x, h), (i) => x[i] + h[i])).toBeLessThan(1e-6);
  });
});
