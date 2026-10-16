import { describe, expect, it } from "vitest";
import { positions, ropeTables } from "../src/kernels/rope";

describe("positions", () => {
  it("puts text first at (0, 0, 0), then image tokens on the latent grid as (0, row, column)", () => {
    expect(positions(2, 2, 3)).toEqual([
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 1],
      [0, 0, 2],
      [0, 1, 0],
      [0, 1, 1],
      [0, 1, 2],
    ]);
  });
});

describe("ropeTables", () => {
  it("gives one angle per pair: 16 + 24 + 24 pairs for axes (32, 48, 48)", () => {
    const { cos, sin, pairs } = ropeTables([[0, 0, 0]]);
    expect(pairs).toBe(64);
    expect(Array.from(cos)).toEqual(new Array(64).fill(1)); // position 0: no rotation
    expect(Array.from(sin)).toEqual(new Array(64).fill(0));
  });

  it("uses theta^(-2j/dim) per axis, in t, h, w order", () => {
    const { cos, sin } = ropeTables([[0, 3, 5]], [32, 48, 48], 1000);
    const angle = (pos: number, j: number, dim: number) => pos * 1000 ** (-(2 * j) / dim);
    expect(sin[0]).toBe(0); // t axis, position 0
    expect(cos[16]).toBeCloseTo(Math.cos(angle(3, 0, 48)), 6); // h axis starts at pair 16
    expect(sin[16 + 5]).toBeCloseTo(Math.sin(angle(3, 5, 48)), 6);
    expect(sin[40 + 23]).toBeCloseTo(Math.sin(angle(5, 23, 48)), 6); // w axis starts at pair 40
  });

  it("refuses axes that don't add up to whole pairs", () => {
    expect(() => ropeTables([[0, 0, 0]], [31, 48, 49])).toThrow(/even/);
  });
});
