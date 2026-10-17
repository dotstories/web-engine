import { expect, it } from "vitest";
import { getDevice } from "../../src/gpu/compute";
import { Graph } from "../../src/gpu/graph";
import { compare } from "../../src/model/compare";
import { linear, type Linear } from "../../src/model/ops";
import { loadLinear } from "../../src/model/weights";
import { matchLora } from "../../src/weights/lora";
import { readHeader } from "../../src/weights/safetensors";
import { makeSafetensors } from "../helpers";
import { e4m3, toFloat32 } from "../../src/weights/fp8";

let seed = 7;
const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32) * 2 - 1;
const floats = (n: number) => Float32Array.from({ length: n }, rand);
/** bf16 bytes (top half of each f32) and the exact values they hold. */
const bf16 = (n: number) => {
  const words = new Uint32Array(floats(n).buffer);
  const bytes = new Uint8Array(new Uint16Array(Array.from(words, (w) => w >>> 16)).buffer);
  return { bytes, values: toFloat32("BF16", bytes) };
};
const fp8 = (n: number) => Uint8Array.from({ length: n }, () => Math.floor((rand() + 1) * 60)); // small, no NaN codes

/** y = x·Wᵀ·scale + s·(x·downᵀ)·upᵀ in float64. */
function reference(x: Float32Array, w: ArrayLike<number>, wScale: number, down: ArrayLike<number>, up: ArrayLike<number>, s: number, m: number, n: number, k: number, r: number) {
  const y = new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    const t = new Float64Array(r);
    for (let j = 0; j < r; j++) for (let c = 0; c < k; c++) t[j] += x[i * k + c] * down[j * k + c];
    for (let o = 0; o < n; o++) {
      let base = 0;
      let delta = 0;
      for (let c = 0; c < k; c++) base += x[i * k + c] * w[o * k + c];
      for (let j = 0; j < r; j++) delta += t[j] * up[o * r + j];
      y[i * n + o] = base * wScale + s * delta;
    }
  }
  return y;
}

it("adds the LoRA update at run time on an fp8 linear (bf16 and f32 LoRA weights)", async () => {
  const [m, n, k, r, s] = [5, 12, 16, 4, 0.8 * 0.5];
  const graph = new Graph(await getDevice());
  try {
    const x = floats(m * k);
    const w = fp8(n * k);
    const base: Linear = { buffer: graph.buffer(new Uint32Array(w.buffer)), dtype: "fp8", scale: 0.25, n, k };
    const wValues = Array.from(w, e4m3);

    const down = bf16(r * k);
    const up = bf16(n * r);
    const bf: Linear = {
      ...base,
      lora: {
        down: { buffer: graph.buffer(new Uint32Array(down.bytes.buffer)), dtype: "bf16", scale: 1, n: r, k },
        up: { buffer: graph.buffer(new Uint32Array(up.bytes.buffer)), dtype: "bf16", scale: s, n, k: r },
      },
    };
    const gotBf = new Float32Array(await graph.read(linear(graph, graph.buffer(x), bf, m), m * n * 4));
    const want = reference(x, wValues, 0.25, down.values, up.values, s, m, n, k, r);
    expect(compare(gotBf, want).relativeError).toBeLessThan(1e-6);

    const f32: Linear = {
      ...base,
      lora: {
        down: { buffer: graph.buffer(down.values), dtype: "f32", scale: 1, n: r, k },
        up: { buffer: graph.buffer(up.values), dtype: "f32", scale: s, n, k: r },
      },
    };
    const gotF32 = new Float32Array(await graph.read(linear(graph, graph.buffer(x), f32, m), m * n * 4));
    expect(compare(gotF32, want).relativeError).toBeLessThan(1e-6);

    // Without the LoRA the result changes: the update is really there.
    const plain = new Float32Array(await graph.read(linear(graph, graph.buffer(x), base, m), m * n * 4));
    expect(compare(plain, want).relativeError).toBeGreaterThan(1e-3);
  } finally {
    graph.destroy();
  }
});

it("rejects a LoRA rank that is not a multiple of 4", async () => {
  const graph = new Graph(await getDevice());
  try {
    const lin = (n: number, k: number, scale = 1): Linear => ({ buffer: graph.buffer(n * k * 4), dtype: "f32", scale, n, k });
    const w: Linear = { ...lin(8, 8), lora: { down: lin(3, 8), up: lin(8, 3) } };
    expect(() => linear(graph, graph.buffer(8 * 4), w, 1)).toThrow("LoRA shapes do not fit the linear layer");
  } finally {
    graph.destroy();
  }
});

it("loads a LoRA with the model weight (kohya names, alpha, rank 3 padded) and applies strength · alpha / rank", async () => {
  const [m, n, k, r, alpha, strength] = [3, 8, 12, 3, 1.5, 0.8];
  const graph = new Graph(await getDevice());
  try {
    const w = bf16(n * k);
    const down = floats(r * k);
    const up = floats(n * r);
    const model = makeSafetensors({ "blocks.0.attn.wq.weight": { dtype: "BF16", shape: [n, k], data: w.bytes } });
    const loraFile = makeSafetensors({
      "lora_unet_blocks_0_attn_wq.lora_down.weight": { dtype: "F32", shape: [r, k], data: new Uint8Array(down.buffer) },
      "lora_unet_blocks_0_attn_wq.lora_up.weight": { dtype: "F32", shape: [n, r], data: new Uint8Array(up.buffer) },
      "lora_unet_blocks_0_attn_wq.alpha": { dtype: "F32", shape: [], data: new Uint8Array(new Float32Array([alpha]).buffer) },
    });
    const st = await readHeader(model);
    const lora = await matchLora(loraFile, await readHeader(loraFile), st);
    const lin = await loadLinear(graph, model, st, "blocks.0.attn.wq", { file: loraFile, lora, strength });

    const x = floats(m * k);
    const got = new Float32Array(await graph.read(linear(graph, graph.buffer(x), lin, m), m * n * 4));
    const want = reference(x, w.values, 1, down, up, (strength * alpha) / r, m, n, k, r);
    expect(compare(got, want).relativeError).toBeLessThan(1e-6);

    const off = await loadLinear(graph, model, st, "blocks.0.attn.wq", { file: loraFile, lora, strength: 0 });
    expect(off.lora).toBeUndefined();
  } finally {
    graph.destroy();
  }
});

it("gives the same result when the LoRA down matmul is split over a long K", async () => {
  const [m, n, k, r, s] = [5, 8, 1024, 4, 0.3]; // kSplitFor → 4 slices
  const graph = new Graph(await getDevice());
  try {
    const x = floats(m * k);
    const w = bf16(n * k);
    const down = bf16(r * k);
    const up = bf16(n * r);
    const lin: Linear = {
      buffer: graph.buffer(new Uint32Array(w.bytes.buffer)),
      dtype: "bf16",
      scale: 1,
      n,
      k,
      lora: {
        down: { buffer: graph.buffer(new Uint32Array(down.bytes.buffer)), dtype: "bf16", scale: 1, n: r, k },
        up: { buffer: graph.buffer(new Uint32Array(up.bytes.buffer)), dtype: "bf16", scale: s, n, k: r },
      },
    };
    const got = new Float32Array(await graph.read(linear(graph, graph.buffer(x), lin, m), m * n * 4));
    expect(compare(got, reference(x, w.values, 1, down.values, up.values, s, m, n, k, r)).relativeError).toBeLessThan(1e-6);
  } finally {
    graph.destroy();
  }
});
