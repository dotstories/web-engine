import { expect, it } from "vitest";
import { timeMatmul } from "../../src/bench/matmul";
import { getDevice } from "../../src/gpu/compute";

it("times an fp8 matmul on the GPU", async () => {
  const ms = await timeMatmul(await getDevice(), { m: 64, n: 128, k: 256 }, 2);
  expect(ms).toBeGreaterThan(0);
  expect(Number.isFinite(ms)).toBe(true);
});

it("times one block's attention and its parts on the GPU", async () => {
  const { timeAttention } = await import("../../src/bench/attention");
  const timings = await timeAttention(await getDevice(), { tokens: 70, heads: 4, kvHeads: 2, dim: 16 });
  expect(timings.map((t) => t.name)).toEqual(["QK (scores)", "softmax", "PV (weighted values)", "attention (all of it)"]);
  for (const t of timings) expect(t.ms).toBeGreaterThan(0);
  expect(timings[0].tflops).toBeGreaterThan(0);
});
