import { describe, expect, it } from "vitest";
import { matmulWgsl, type Weights } from "../src/kernels/matmul";
import { tunedWgsl, TUNE_CONFIGS } from "../src/kernels/matmul-tuned";

const kernels = (["fp8", "bf16", "f32"] as Weights[]).flatMap((w) => [
  matmulWgsl(w),
  matmulWgsl(w, "rows", { splitK: true }),
  matmulWgsl(w, "rows", { accumulate: true }),
]);

describe("matmul WGSL", () => {
  it("multiply-adds with fma() (on D3D12 'acc += a * b' is a separate multiply and add: ~1.5x slower)", () => {
    for (const code of [...kernels, matmulWgsl("f32", "conv3x3"), ...TUNE_CONFIGS.map(tunedWgsl)]) {
      expect(code).not.toMatch(/acc\d+ \+= /);
      expect(code).toMatch(/acc0 = fma\(/);
    }
  });

  it("unrolls the 16 k of a step: 16 × 16 fma per step", () => {
    for (const code of kernels) expect(code.match(/= fma\(/g)).toHaveLength(256);
  });
});
