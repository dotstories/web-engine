import { describe, expect, it } from "vitest";
import { getDevice, runCompute } from "../../src/gpu/compute";
import { matmulBf16, matmulF32, matmulFp8, matmulParams, matmulWgsl, matmulWorkgroups } from "../../src/kernels/matmul";
import { Graph } from "../../src/gpu/graph";
import { kSplitFor } from "../../src/model/ops";
import { toFloat32 } from "../../src/weights/fp8";
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

  it("refuses K, strides or offsets that are not a multiple of 4", () => {
    expect(() => matmulParams({ m: 1, n: 1, k: 6 }, 1)).toThrow(/K must be a multiple of 4/);
    expect(() => matmulParams({ m: 1, n: 1, k: 4 }, 1, { xStride: 6 })).toThrow(/xStride/);
    expect(() => matmulParams({ m: 1, n: 1, k: 4 }, 1, { wOffset: 2 })).toThrow(/wOffset/);
  });
});

describe("matmulF32", () => {
  const floats = (n: number, seed: number) => {
    const r = rng(seed);
    return Float32Array.from({ length: n }, () => r() * 2 - 1);
  };

  it.each([
    [37, 70, 100],
    [130, 65, 36],
  ])("matches the CPU reference for M=%i N=%i K=%i", async (m, n, k) => {
    const [x, w] = [floats(m * k, 1), floats(n * k, 2)];
    const gpu = await matmulF32(await getDevice(), x, w, 0.5, { m, n, k });
    const cpu = new Float64Array(m * n);
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) for (let t = 0; t < k; t++) cpu[i * n + j] += x[i * k + t] * w[j * k + t] * 0.5;
    expect(relError(gpu, cpu)).toBeLessThan(1e-5);
  });

  it("picks one head out of [tokens, heads, dim] with strides and offsets (Q·Kᵀ of head 2)", async () => {
    const [tokens, heads, kvHeads, dim] = [20, 4, 2, 128];
    const q = floats(tokens * heads * dim, 3);
    const k = floats(tokens * kvHeads * dim, 4);
    const [h, g] = [2, 1]; // query head 2 uses kv head 1
    const scores = await matmulF32(await getDevice(), q, k, 1, { m: tokens, n: tokens, k: dim }, {
      xStride: heads * dim, xOffset: h * dim, wStride: kvHeads * dim, wOffset: g * dim, yStride: 24,
    }, tokens * 24);
    let err = 0;
    for (let i = 0; i < tokens; i++)
      for (let j = 0; j < tokens; j++) {
        let s = 0;
        for (let d = 0; d < dim; d++) s += q[(i * heads + h) * dim + d] * k[(j * kvHeads + g) * dim + d];
        err = Math.max(err, Math.abs(scores[i * 24 + j] - s));
      }
    expect(err).toBeLessThan(1e-4);
  });
});

describe("matmulBf16", () => {
  it.each([
    [37, 70, 100],
    [8, 1536, 6144], // a Krea 2 shape
  ])("matches the CPU reference for M=%i N=%i K=%i", async (m, n, k) => {
    const r = rng(9);
    const x = Float32Array.from({ length: m * k }, () => r() * 2 - 1);
    // bf16 = the top half of an f32
    const f32 = Float32Array.from({ length: n * k }, () => r() * 2 - 1);
    const w = new Uint8Array(Uint16Array.from(new Uint32Array(f32.buffer), (u) => u >>> 16).buffer);
    const wf = toFloat32("BF16", w);
    const gpu = await matmulBf16(await getDevice(), x, w, 0.5, { m, n, k });
    const cpu = new Float64Array(m * n);
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) for (let t = 0; t < k; t++) cpu[i * n + j] += x[i * k + t] * wf[j * k + t] * 0.5;
    expect(relError(gpu, cpu)).toBeLessThan(1e-5);
  });
});

describe("split-K matmul", () => {
  it("each workgroup layer sums its own K slice into its own partial (fp8, uneven slices, scale)", async () => {
    const device = await getDevice();
    const [m, n, k, split, scale] = [5, 12, 200, 3, 0.5]; // slices of 80, 80, 40
    const { x, w } = data(m, n, k, 3);
    const packed = new Uint8Array(Math.ceil(w.length / 4) * 4);
    packed.set(w);
    const [partials] = await runCompute(device, {
      code: matmulWgsl("fp8", "rows", { splitK: true }),
      inputs: [matmulParams({ m, n, k }, scale, { kSplit: split }), x, packed],
      outputs: [split * m * n * 4],
      workgroups: [...matmulWorkgroups({ m, n, k }), split],
    });
    const parts = new Float32Array(partials);
    const sum = Array.from({ length: m * n }, (_, i) => parts[i] + parts[m * n + i] + parts[2 * m * n + i]);
    const want = reference(x, w, scale, m, n, k);
    for (let i = 0; i < want.length; i++) expect(sum[i]).toBeCloseTo(want[i], 3);
    // the first slice alone is not the whole result
    expect(Math.abs(parts[0] - want[0])).toBeGreaterThan(1e-3);
  });

  it("splits only narrow outputs with a long K", () => {
    expect(kSplitFor(1088, 64, 6144)).toBe(24); // 9 tiles → 29 wanted, capped at 6144 / 256
    expect(kSplitFor(4160, 64, 16384)).toBe(8); // 33 tiles → 8
    expect(kSplitFor(4160, 6144, 6144)).toBe(1); // already fills the GPU
    expect(kSplitFor(5, 12, 16)).toBe(1); // K too short
  });
});

it("accumulate mode adds to y instead of overwriting it", async () => {
  const device = await getDevice();
  const graph = new Graph(device);
  try {
    const [m, n, k, scale] = [3, 6, 8, 0.25];
    const x = Float32Array.from({ length: m * k }, (_, i) => (i % 5) - 2);
    const w = Float32Array.from({ length: n * k }, (_, i) => ((i * 3) % 7) - 3);
    const start = Float32Array.from({ length: m * n }, (_, i) => i);
    const y = graph.buffer(start);
    graph.run(matmulWgsl("f32", "rows", { accumulate: true }), [graph.buffer(matmulParams({ m, n, k }, scale)), graph.buffer(x), graph.buffer(w), y], matmulWorkgroups({ m, n, k }));
    const got = new Float32Array(await graph.read(y));
    const plain = await matmulF32(device, x, w, scale, { m, n, k });
    expect(Array.from(got)).toEqual(Array.from(plain, (v, i) => v + start[i]));
  } finally {
    graph.destroy();
  }
});
