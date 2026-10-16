import { describe, expect, it } from "vitest";
import { compare } from "../src/model/compare";

describe("compare", () => {
  it("is perfect for equal arrays", () => {
    expect(compare([1, 2, 3], [1, 2, 3])).toEqual({ cosine: 1, maxError: 0, relativeError: 0 });
  });

  it("measures direction, largest error and relative size of the error", () => {
    const r = compare([1, 0], [0, 1]);
    expect(r.cosine).toBe(0);
    expect(r.maxError).toBe(1);
    expect(r.relativeError).toBeCloseTo(Math.SQRT2);
  });

  it("refuses arrays of different length", () => {
    expect(() => compare([1], [1, 2])).toThrow(/Lengths differ/);
  });
});
