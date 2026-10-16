import { describe, expect, it } from "vitest";
import { getDevice } from "../../src/gpu/compute";
import { elementwiseWorkgroups, gatedAdd, sigmoidGate, swiglu } from "../../src/kernels/elementwise";

function rng(seed: number) {
  return () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32 - 0.5;
}
const random = (n: number, seed: number, size = 16) => {
  const r = rng(seed);
  return Float32Array.from({ length: n }, () => r() * size);
};
const sigmoid = (v: number) => 1 / (1 + Math.exp(-v));
const maxError = (a: Float32Array, ref: (i: number) => number) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - ref(i))), 0);

describe("elementwise kernels", () => {
  const n = 1000;

  it("swiglu: silu(gate) · up", async () => {
    const [g, u] = [random(n, 1), random(n, 2)];
    const gpu = await swiglu(await getDevice(), g, u);
    expect(maxError(gpu, (i) => g[i] * sigmoid(g[i]) * u[i])).toBeLessThan(1e-4);
  });

  it("sigmoidGate: a · sigmoid(gate)", async () => {
    const [a, g] = [random(n, 3), random(n, 4)];
    const gpu = await sigmoidGate(await getDevice(), a, g);
    expect(maxError(gpu, (i) => a[i] * sigmoid(g[i]))).toBeLessThan(1e-5);
  });

  it("gatedAdd: x + g[column] · h, with one gate value per column", async () => {
    const d = 40;
    const [x, h, g] = [random(n, 5), random(n, 6), random(d, 7, 2)];
    const gpu = await gatedAdd(await getDevice(), x, h, g);
    expect(maxError(gpu, (i) => x[i] + g[i % d] * h[i])).toBeLessThan(1e-5);
  });

  it("splits big tensors across two dispatch dimensions", () => {
    expect(elementwiseWorkgroups(1000)).toEqual([4, 1]);
    expect(elementwiseWorkgroups(4608 * 16384)).toEqual([65535, 5]);
  });
});
