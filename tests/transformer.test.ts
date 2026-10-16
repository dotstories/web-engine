import { describe, expect, it } from "vitest";
import { patchify, timestepEmbedding, unpatchify } from "../src/model/transformer";

describe("timestepEmbedding", () => {
  it("is [cos, sin] of 1000·t·10000^(-i/128), cos first (Flux/ComfyUI)", () => {
    const e = timestepEmbedding(0.7);
    expect(e.length).toBe(256);
    expect(e[0]).toBeCloseTo(Math.cos(700), 5);
    expect(e[128]).toBeCloseTo(Math.sin(700), 5);
    expect(e[127]).toBeCloseTo(Math.cos(700 * 10000 ** (-127 / 128)), 5);
  });

  it("gives cos 1, sin 0 at t = 0", () => {
    const e = timestepEmbedding(0);
    expect(Array.from(e.slice(0, 128))).toEqual(new Array(128).fill(1));
    expect(Array.from(e.slice(128))).toEqual(new Array(128).fill(0));
  });
});

describe("patchify", () => {
  it("turns 2×2 patches into 64 features in (channel, row, column) order", () => {
    const [h, w] = [4, 6];
    const latent = Float32Array.from({ length: 16 * h * w }, (_, i) => i);
    const tokens = patchify(latent, h, w);
    expect(tokens.length).toBe((h / 2) * (w / 2) * 64);
    // token (row 1, col 2), channel 3, patch pixel (1, 0) = latent[3, 2 + 1, 4 + 0]
    expect(tokens[(1 * 3 + 2) * 64 + 3 * 4 + 1 * 2 + 0]).toBe(latent[(3 * h + 3) * w + 4]);
  });

  it("unpatchify undoes it", () => {
    const latent = Float32Array.from({ length: 16 * 8 * 8 }, (_, i) => Math.sin(i));
    expect(Array.from(unpatchify(patchify(latent, 8, 8), 8, 8))).toEqual(Array.from(latent));
  });
});
