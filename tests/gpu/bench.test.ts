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

it("profiles one block op by op and as a whole", async () => {
  const { profileBlock } = await import("../../src/bench/block");
  const cfg = { hidden: 64, mlp: 96, heads: 4, kvHeads: 2, headDim: 16, eps: 1e-5 };
  const p = await profileBlock(await getDevice(), { textTokens: 3, rows: 2, columns: 3 }, cfg);
  expect(p.tokens).toBe(9);
  expect(p.ops.map((o) => o.name)).toContain("attention");
  expect(p.ops.filter((o) => o.kind === "linear")).toHaveLength(8);
  for (const o of p.ops) expect(o.ms).toBeGreaterThan(0);
  expect(p.block).toBeGreaterThan(0);
});
