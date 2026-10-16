import { describe, expect, it } from "vitest";
import { eulerStep, gaussian, sigmas } from "../src/model/sampler";

describe("sigmas", () => {
  it("has 8 steps from 1 to near 0, then 0, always going down", () => {
    const s = sigmas();
    expect(s.length).toBe(9);
    expect(s[0]).toBeCloseTo(1, 10); // shift keeps 1 at 1
    expect(s[8]).toBe(0);
    for (let i = 1; i < s.length; i++) expect(s[i]).toBeLessThan(s[i - 1]);
  });

  it("shifts linspace(1, 1/8, 8) with e^mu / (e^mu + 1/σ − 1)", () => {
    const e = Math.exp(1.15);
    expect(sigmas()[7]).toBeCloseTo(e / (e + 8 - 1), 10); // last raw sigma is 1/8
    expect(sigmas(1)).toEqual([1, 0]);
  });
});

describe("eulerStep", () => {
  it("moves x along the velocity by the sigma step", () => {
    expect(Array.from(eulerStep(new Float32Array([1, 2]), new Float32Array([4, -2]), 1, 0.75))).toEqual([0, 2.5]);
  });
});

describe("gaussian", () => {
  it("is repeatable and roughly standard normal", () => {
    const a = gaussian(20000, 42);
    expect(Array.from(a.slice(0, 5))).toEqual(Array.from(gaussian(5, 42)));
    const mean = a.reduce((s, v) => s + v, 0) / a.length;
    const std = Math.sqrt(a.reduce((s, v) => s + (v - mean) ** 2, 0) / a.length);
    expect(Math.abs(mean)).toBeLessThan(0.03);
    expect(Math.abs(std - 1)).toBeLessThan(0.03);
  });
});
