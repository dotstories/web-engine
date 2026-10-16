import { expect, it } from "vitest";
import { timeMatmul } from "../../src/bench/matmul";
import { getDevice } from "../../src/gpu/compute";

it("times an fp8 matmul on the GPU", async () => {
  const ms = await timeMatmul(await getDevice(), { m: 64, n: 128, k: 256 }, 2);
  expect(ms).toBeGreaterThan(0);
  expect(Number.isFinite(ms)).toBe(true);
});
