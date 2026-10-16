import { describe, expect, it } from "vitest";
import { getDevice } from "../../src/gpu/compute";
import { rmsnorm, rmsnormWorkgroups } from "../../src/kernels/rmsnorm";

function rng(seed: number) {
  return () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32 - 0.5;
}
const random = (n: number, r: () => number, size = 2) => Float32Array.from({ length: n }, () => r() * size);

function reference(x: Float32Array, w: Float32Array, scale: Float32Array, shift: Float32Array, eps: number, offset: number) {
  const d = w.length;
  const out = new Float64Array(x.length);
  for (let row = 0; row < x.length / d; row++) {
    let sum = 0;
    for (let i = 0; i < d; i++) sum += x[row * d + i] ** 2;
    const inv = 1 / Math.sqrt(sum / d + eps);
    for (let i = 0; i < d; i++) out[row * d + i] = x[row * d + i] * inv * (offset + w[i]) * (1 + scale[i]) + shift[i];
  }
  return out;
}

const maxError = (a: Float32Array, b: Float64Array) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);

describe("rmsnorm", () => {
  it("normalizes Krea 2 rows (d = 6144) with modulation, zero-centered weight", async () => {
    const r = rng(1);
    const d = 6144;
    const [x, w, scale, shift] = [random(3 * d, r, 8), random(d, r, 0.2), random(d, r), random(d, r)];
    const gpu = await rmsnorm(await getDevice(), x, w, { scale, shift });
    expect(maxError(gpu, reference(x, w, scale, shift, 1e-5, 1))).toBeLessThan(1e-5);
  });

  it("works as plain RMSNorm per attention head (d = 128, offset 0, no modulation)", async () => {
    const r = rng(2);
    const d = 128;
    const [x, w] = [random(100 * d, r), random(d, r)];
    const zeros = new Float32Array(d);
    const gpu = await rmsnorm(await getDevice(), x, w, { offset: 0 });
    expect(maxError(gpu, reference(x, w, zeros, zeros, 1e-5, 0))).toBeLessThan(1e-5);
  });

  it("handles rows shorter than the workgroup and odd sizes", async () => {
    const r = rng(3);
    const [x, w] = [random(5 * 37, r), random(37, r)];
    const zeros = new Float32Array(37);
    const gpu = await rmsnorm(await getDevice(), x, w);
    expect(maxError(gpu, reference(x, w, zeros, zeros, 1e-5, 1))).toBeLessThan(1e-5);
  });

  it("splits many rows across two dispatch dimensions", () => {
    expect(rmsnormWorkgroups(10)).toEqual([10, 1]);
    expect(rmsnormWorkgroups(4608 * 48)).toEqual([65535, 4]);
  });
});
