import { describe, expect, it } from "vitest";
import { BLOCK_MATMULS, KREA, blockParams, estimateImage, tokensFor } from "../src/bench/estimate";

describe("Krea 2 sizes", () => {
  it("has 434M weights per block and 12.2B in all 28 blocks", () => {
    expect(blockParams()).toBe(434_110_464);
    expect(blockParams() * KREA.blocks).toBe(12_155_092_992);
  });

  it("counts tokens: 16×16 pixels per image token (VAE 8 × patch 2) plus 512 text tokens", () => {
    expect(tokensFor(1024, 1024)).toBe(4096 + 512);
    expect(tokensFor(1920, 1088)).toBe(120 * 68 + 512);
  });
});

describe("estimateImage", () => {
  const tokens = 1000;
  // every matmul at exactly 10 TFLOPS: ms = 2·M·N·K / 1e10
  const timings = BLOCK_MATMULS.map((s) => ({ ...s, ms: (2 * tokens * s.n * s.k) / 1e10 }));

  it("adds a block's matmuls and measures the speed reached", () => {
    const e = estimateImage(timings, tokens);
    expect(e.tflops).toBeCloseTo(10);
    expect(e.matmulMs).toBeCloseTo((2 * tokens * blockParams()) / 1e10);
  });

  it("adds attention at the same speed and scales to 28 blocks × 8 steps", () => {
    const e = estimateImage(timings, tokens);
    const attentionMs = (4 * tokens ** 2 * KREA.hidden) / 1e10;
    expect(e.blockMs).toBeCloseTo(e.matmulMs + attentionMs);
    expect(e.imageSeconds).toBeCloseTo((e.blockMs * KREA.blocks * KREA.steps) / 1000);
  });
});
