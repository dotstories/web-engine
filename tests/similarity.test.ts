import { describe, expect, it } from "vitest";
import { psnr, ssim } from "../src/check/similarity";

const [W, H] = [64, 48];

/** A seeded test picture: soft gradients plus some edges. */
function picture(seed = 1): Uint8ClampedArray {
  const out = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const k = (y * W + x) * 4;
      out[k] = (x * 4 + seed * 17) % 256;
      out[k + 1] = (y * 5) % 256;
      out[k + 2] = (x + y) % 16 < 8 ? 40 : 200;
      out[k + 3] = 255;
    }
  }
  return out;
}

function noisy(src: Uint8ClampedArray, amount: number): Uint8ClampedArray {
  let s = 7;
  return src.map((v, i) => (i % 4 === 3 ? v : v + (((s = (s * 1103515245 + 12345) >>> 0) % (2 * amount + 1)) - amount)));
}

describe("psnr", () => {
  it("is Infinity for the same image and drops as noise grows", () => {
    const a = picture();
    expect(psnr(a, a)).toBe(Infinity);
    const small = psnr(a, noisy(a, 2));
    const big = psnr(a, noisy(a, 30));
    expect(small).toBeGreaterThan(40);
    expect(big).toBeLessThan(small);
  });

  it("ignores alpha", () => {
    const a = picture();
    const b = a.map((v, i) => (i % 4 === 3 ? 0 : v));
    expect(psnr(a, b)).toBe(Infinity);
  });

  it("rejects different sizes", () => {
    expect(() => psnr(picture(), new Uint8ClampedArray(4))).toThrow("same size");
  });
});

describe("ssim", () => {
  it("is 1 for the same image, high for slight noise, low for an unrelated image", () => {
    const a = picture();
    expect(ssim(a, a, W, H)).toBeCloseTo(1, 10);
    expect(ssim(a, noisy(a, 2), W, H)).toBeGreaterThan(0.95);
    const inverted = a.map((v, i) => (i % 4 === 3 ? v : 255 - v));
    expect(ssim(a, inverted, W, H)).toBeLessThan(0.3);
  });

  it("goes down as the difference grows", () => {
    const a = picture();
    expect(ssim(a, noisy(a, 30), W, H)).toBeLessThan(ssim(a, noisy(a, 5), W, H));
  });
});
