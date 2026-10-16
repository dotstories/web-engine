import { describe, expect, it } from "vitest";
import { getDevice } from "../../src/gpu/compute";
import { silu } from "../../src/kernels/elementwise";
import { upsample2x } from "../../src/kernels/image";
import { conv3x3 } from "../../src/kernels/matmul";

let seed = 5;
const random = (n: number) => Float32Array.from({ length: n }, () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32) * 2 - 1);

describe("conv3x3", () => {
  it("matches a direct 3×3 convolution with zero padding (channel-last)", async () => {
    const [h, w, cin, cout] = [5, 7, 8, 12];
    const x = random(h * w * cin);
    const weight = random(cout * 9 * cin); // [cout, ky, kx, cin]
    const gpu = await conv3x3(await getDevice(), x, weight, h, w, cin, cout);
    let err = 0;
    for (let y = 0; y < h; y++)
      for (let xx = 0; xx < w; xx++)
        for (let o = 0; o < cout; o++) {
          let s = 0;
          for (let ky = 0; ky < 3; ky++)
            for (let kx = 0; kx < 3; kx++) {
              const [sy, sx] = [y + ky - 1, xx + kx - 1];
              if (sy < 0 || sx < 0 || sy >= h || sx >= w) continue;
              for (let c = 0; c < cin; c++) s += x[(sy * w + sx) * cin + c] * weight[((o * 3 + ky) * 3 + kx) * cin + c];
            }
          err = Math.max(err, Math.abs(gpu[(y * w + xx) * cout + o] - s));
        }
    expect(err).toBeLessThan(1e-5);
  });
});

describe("upsample2x", () => {
  it("repeats every pixel 2×2, channel by channel", async () => {
    const [h, w, c] = [2, 3, 2];
    const x = Float32Array.from({ length: h * w * c }, (_, i) => i);
    const gpu = await upsample2x(await getDevice(), x, h, w, c);
    for (let y = 0; y < 2 * h; y++)
      for (let xx = 0; xx < 2 * w; xx++)
        for (let ch = 0; ch < c; ch++) expect(gpu[(y * 2 * w + xx) * c + ch]).toBe(x[((y >> 1) * w + (xx >> 1)) * c + ch]);
  });
});

describe("silu", () => {
  it("is x · sigmoid(x)", async () => {
    const x = Float32Array.from([...random(100), -50, 50, 0]);
    const gpu = await silu(await getDevice(), x);
    expect(gpu.reduce((m, v, i) => Math.max(m, Math.abs(v - x[i] / (1 + Math.exp(-x[i])))), 0)).toBeLessThan(1e-6);
  });
});
