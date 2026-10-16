import { describe, expect, it } from "vitest";
import { getDevice } from "../../src/gpu/compute";
import { matmulFp8, matmulParams } from "../../src/kernels/matmul";
import { e4m3 } from "../../src/weights/fp8";

/** Small deterministic random numbers (same every run). */
function rng(seed: number) {
  return () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
}

function data(m: number, n: number, k: number, seed = 1) {
  const r = rng(seed);
  const x = Float32Array.from({ length: m * k }, () => r() * 2 - 1);
  // any fp8 byte except the NaN codes (0x7f, 0xff)
  const w = Uint8Array.from({ length: n * k }, () => {
    const b = Math.floor(r() * 256);
    return (b & 0x7f) === 0x7f ? b - 1 : b;
  });
  return { x, w };
}

/** CPU reference in float64. */
function reference(x: Float32Array, w: Uint8Array, scale: number, m: number, n: number, k: number) {
  const y = new Float64Array(m * n);
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let t = 0; t < k; t++) s += x[i * k + t] * e4m3(w[j * k + t]);
      y[i * n + j] = s * scale;
    }
  return y;
}

/** Largest error relative to the size of the result. */
function relError(gpu: Float32Array, cpu: Float64Array) {
  let err = 0;
  let max = 0;
  for (let i = 0; i < cpu.length; i++) {
    err = Math.max(err, Math.abs(gpu[i] - cpu[i]));
    max = Math.max(max, Math.abs(cpu[i]));
  }
  return err / max;
}

describe("matmulFp8", () => {
  it.each([
    [1, 1, 4],
    [37, 70, 100], // sizes that don't fill the tiles
    [64, 64, 64],
    [130, 65, 36],
  ])("matches the CPU reference for M=%i N=%i K=%i", async (m, n, k) => {
    const { x, w } = data(m, n, k);
    const gpu = await matmulFp8(await getDevice(), x, w, 0.002, { m, n, k });
    expect(relError(gpu, reference(x, w, 0.002, m, n, k))).toBeLessThan(1e-5);
  });

  it("works with a Krea 2 layer shape (K = 6144, N = 1536)", async () => {
    const [m, n, k] = [8, 1536, 6144];
    const { x, w } = data(m, n, k, 7);
    const gpu = await matmulFp8(await getDevice(), x, w, 0.0018833705689758062, { m, n, k });
    expect(relError(gpu, reference(x, w, 0.0018833705689758062, m, n, k))).toBeLessThan(1e-5);
  });

  it("refuses K that is not a multiple of 4", () => {
    expect(() => matmulParams({ m: 1, n: 1, k: 6 }, 1)).toThrow(/multiple of 4/);
  });
});
