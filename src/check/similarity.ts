/**
 * How close two images are: used to check that the browser makes the same picture as ComfyUI.
 * Both take RGBA bytes of the same size (as from canvas getImageData).
 */

/** Peak signal-to-noise ratio over R, G and B, in dB (Infinity when identical). ≥ 40 dB looks the same. */
export function psnr(a: Uint8ClampedArray, b: Uint8ClampedArray): number {
  if (a.length !== b.length) throw new Error("Images must be the same size");
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    if (i % 4 === 3) continue; // alpha
    const d = a[i] - b[i];
    sum += d * d;
  }
  const mse = sum / ((a.length / 4) * 3);
  return mse === 0 ? Infinity : 10 * Math.log10((255 * 255) / mse);
}

function luma(rgba: Uint8ClampedArray): Float32Array {
  const out = new Float32Array(rgba.length / 4);
  for (let i = 0; i < out.length; i++) out[i] = 0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2];
  return out;
}

const C1 = (0.01 * 255) ** 2;
const C2 = (0.03 * 255) ** 2;

/**
 * Structural similarity of the brightness, the mean over 8×8 blocks (1 = identical, ~0 = unrelated).
 * Compares shapes and contrast, so it is forgiving of tiny colour shifts that PSNR punishes.
 */
export function ssim(a: Uint8ClampedArray, b: Uint8ClampedArray, width: number, height: number, block = 8): number {
  if (a.length !== b.length || a.length !== width * height * 4) throw new Error("Images must be the same size");
  const [x, y] = [luma(a), luma(b)];
  let total = 0;
  let count = 0;
  for (let by = 0; by + block <= height; by += block) {
    for (let bx = 0; bx + block <= width; bx += block) {
      let [sx, sy, sxx, syy, sxy] = [0, 0, 0, 0, 0];
      for (let j = 0; j < block; j++) {
        for (let i = 0; i < block; i++) {
          const k = (by + j) * width + bx + i;
          sx += x[k];
          sy += y[k];
          sxx += x[k] * x[k];
          syy += y[k] * y[k];
          sxy += x[k] * y[k];
        }
      }
      const n = block * block;
      const [mx, my] = [sx / n, sy / n];
      const [vx, vy, cov] = [sxx / n - mx * mx, syy / n - my * my, sxy / n - mx * my];
      total += ((2 * mx * my + C1) * (2 * cov + C2)) / ((mx * mx + my * my + C1) * (vx + vy + C2));
      count++;
    }
  }
  return total / count;
}
