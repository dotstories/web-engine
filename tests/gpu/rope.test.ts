import { expect, it } from "vitest";
import { getDevice } from "../../src/gpu/compute";
import { positions, rope, ropeTables } from "../../src/kernels/rope";

it("rotates each pair of each head by its token's angles", async () => {
  const heads = 3;
  const pos = positions(2, 3, 4); // 2 text + 12 image tokens
  const tables = ropeTables(pos);
  let seed = 1;
  const x = Float32Array.from({ length: pos.length * heads * 128 }, () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32) * 2 - 1);

  const gpu = await rope(await getDevice(), x, heads, tables);

  let err = 0;
  for (let t = 0; t < pos.length; t++)
    for (let h = 0; h < heads; h++)
      for (let j = 0; j < 64; j++) {
        const i = ((t * heads + h) * 64 + j) * 2;
        const [c, s] = [tables.cos[t * 64 + j], tables.sin[t * 64 + j]];
        err = Math.max(err, Math.abs(gpu[i] - (x[i] * c - x[i + 1] * s)), Math.abs(gpu[i + 1] - (x[i + 1] * c + x[i] * s)));
      }
  expect(err).toBeLessThan(1e-6);
  expect(Array.from(gpu.slice(0, 2 * heads * 128))).toEqual(Array.from(x.slice(0, 2 * heads * 128))); // text: unchanged
});

it("rotates (j, j + 64) pairs for the text encoder (Qwen's rotate_half)", async () => {
  const heads = 2;
  const pos = [[0], [1], [5], [40]]; // token positions, one axis of 128 dims
  const tables = ropeTables(pos, [128], 5e6);
  let s = 7;
  const x = Float32Array.from({ length: pos.length * heads * 128 }, () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32) * 2 - 1);
  const gpu = await rope(await getDevice(), x, heads, tables, "half");
  let err = 0;
  for (let t = 0; t < pos.length; t++)
    for (let h = 0; h < heads; h++)
      for (let j = 0; j < 64; j++) {
        const base = (t * heads + h) * 128;
        const [c, sn] = [Math.cos(pos[t][0] * 5e6 ** (-(2 * j) / 128)), Math.sin(pos[t][0] * 5e6 ** (-(2 * j) / 128))];
        err = Math.max(err, Math.abs(gpu[base + j] - (x[base + j] * c - x[base + j + 64] * sn)), Math.abs(gpu[base + j + 64] - (x[base + j + 64] * c + x[base + j] * sn)));
      }
  expect(err).toBeLessThan(1e-5);
});
