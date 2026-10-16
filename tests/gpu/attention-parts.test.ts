import { describe, expect, it } from "vitest";
import { getDevice } from "../../src/gpu/compute";
import { padTo4, softmax, transposeHeads } from "../../src/kernels/attention-parts";

let seed = 1;
const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32) * 2 - 1;

describe("softmax", () => {
  it("normalizes each row over its valid columns and zeroes the padding", async () => {
    const [rows, cols, stride] = [5, 1001, 1004];
    const x = Float32Array.from({ length: rows * stride }, () => rand() * 30); // big values: needs the max trick
    const gpu = await softmax(await getDevice(), x, rows, cols, stride);
    let err = 0;
    for (let r = 0; r < rows; r++) {
      const row = Array.from(x.slice(r * stride, r * stride + cols));
      const m = Math.max(...row);
      const sum = row.reduce((s, v) => s + Math.exp(v - m), 0);
      row.forEach((v, i) => (err = Math.max(err, Math.abs(gpu[r * stride + i] - Math.exp(v - m) / sum))));
      expect(Array.from(gpu.slice(r * stride + cols, (r + 1) * stride))).toEqual([0, 0, 0]);
    }
    expect(err).toBeLessThan(1e-6);
  });
});

describe("transposeHeads", () => {
  it("turns V [tokens, heads, dim] into Vᵀ [heads, dim, tokens padded to 4]", async () => {
    const [tokens, heads, dim] = [7, 3, 16];
    const v = Float32Array.from({ length: tokens * heads * dim }, (_, i) => i);
    const stride = padTo4(tokens);
    const gpu = await transposeHeads(await getDevice(), v, tokens, heads, dim);
    expect(gpu.length).toBe(heads * dim * stride);
    for (let h = 0; h < heads; h++)
      for (let d = 0; d < dim; d++)
        for (let t = 0; t < stride; t++) {
          const want = t < tokens ? v[(t * heads + h) * dim + d] : 0;
          if (gpu[(h * dim + d) * stride + t] !== want) throw new Error(`mismatch at head ${h}, dim ${d}, token ${t}`);
        }
  });
});
