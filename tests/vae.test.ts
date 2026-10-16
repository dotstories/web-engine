import { describe, expect, it } from "vitest";
import { LATENT_MEAN, LATENT_STD, channelFirst, channelLast, convWeight, toRgba, vaeInput } from "../src/model/vae";

describe("channel layout", () => {
  it("channelLast and channelFirst undo each other", () => {
    const x = Float32Array.from({ length: 3 * 4 }, (_, i) => i); // [3, 2, 2]
    expect(Array.from(channelLast(x, 3, 4))).toEqual([0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]);
    expect(Array.from(channelFirst(channelLast(x, 3, 4), 3, 4))).toEqual(Array.from(x));
  });
});

describe("vaeInput", () => {
  it("applies the Wan 2.1 latent format per channel: x · std + mean", () => {
    const latent = new Float32Array(16 * 2).fill(1); // [16, 1, 2]
    const z = vaeInput(latent, 2);
    expect(z[0]).toBeCloseTo(LATENT_STD[0] + LATENT_MEAN[0], 5);
    expect(z[15]).toBeCloseTo(LATENT_STD[15] + LATENT_MEAN[15], 5);
    expect(z[16]).toBeCloseTo(z[0], 5); // second pixel, channel 0
  });
});

describe("toRgba", () => {
  it("maps [-1, 1] to 0..255, clamps, and sets alpha", () => {
    expect(Array.from(toRgba(Float32Array.from([-1, 0, 1, -5, 5, 0.5]), 2))).toEqual([0, 128, 255, 255, 0, 255, 191, 255]);
  });
});

describe("convWeight", () => {
  it("takes the last time slice of a causal 3D kernel and orders it [cout, ky, kx, cin]", () => {
    const [cout, cin, kt, k] = [2, 3, 3, 3];
    const w = Float32Array.from({ length: cout * cin * kt * k * k }, (_, i) => i);
    const out = convWeight(w, [cout, cin, kt, k, k]);
    // out[o, y, x, c] = w[o, c, 2, y, x]
    expect(out[((1 * 3 + 2) * 3 + 0) * cin + 1]).toBe(w[(((1 * cin + 1) * kt + 2) * k + 2) * k + 0]);
    expect(out.length).toBe(cout * 9 * cin);
  });

  it("handles 2D convs and 1×1 convs", () => {
    expect(Array.from(convWeight(Float32Array.from([1, 2, 3, 4]), [2, 2, 1, 1]))).toEqual([1, 2, 3, 4]);
    expect(convWeight(new Float32Array(2 * 4 * 9), [2, 4, 3, 3]).length).toBe(72);
  });
});
