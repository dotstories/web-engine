import { describe, expect, it } from "vitest";
import { IMAGE_SIZES, parseSize, resolution } from "../src/pipeline/sizes";

describe("resolution (ComfyUI ResolutionSelector)", () => {
  it("gives 1928 × 1088 for 16:9 at 2 MP, multiple 8 (the Dark Fantasy Kit)", () => {
    expect(resolution([16, 9], 2)).toEqual({ width: 1928, height: 1088 });
  });

  it("uses binary megapixels: 1 MP square is 1024 × 1024", () => {
    expect(resolution([1, 1], 1)).toEqual({ width: 1024, height: 1024 });
    expect(resolution([1, 1], 0.25)).toEqual({ width: 512, height: 512 });
  });

  it("rounds each side to the nearest multiple", () => {
    const { width, height } = resolution([21, 9], 2, 64);
    expect(width % 64).toBe(0);
    expect(height % 64).toBe(0);
  });
});

describe("IMAGE_SIZES", () => {
  it("are all multiples of 8 and include the Kit size", () => {
    for (const s of IMAGE_SIZES) {
      expect(s.width % 8).toBe(0);
      expect(s.height % 8).toBe(0);
    }
    expect(IMAGE_SIZES.some((s) => s.width === 1928 && s.height === 1088)).toBe(true);
  });
});

describe("parseSize", () => {
  it("reads 'W×H' values", () => {
    expect(parseSize("1928x1088")).toEqual({ width: 1928, height: 1088 });
  });

  it("rejects sizes that are not multiples of 8", () => {
    expect(() => parseSize("1000x1001")).toThrow(/multiple of 8/);
    expect(() => parseSize("big")).toThrow();
  });
});
