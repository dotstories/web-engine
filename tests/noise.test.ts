import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { MT19937, torchRandn } from "../src/model/noise";

type Case = { n: number; seed: number; values: Record<string, number> };
const fixture: { cases: Case[] } = JSON.parse(readFileSync(new URL("fixtures/torch-randn.json", import.meta.url), "utf8"));

const round4 = (a: ArrayLike<number>) => Array.from(a, (x) => Math.round(x * 1e4) / 1e4);

describe("MT19937", () => {
  it("matches the reference MT19937 output for seed 5489 (10000th number = 4123659995)", () => {
    const mt = new MT19937(5489);
    let last = 0;
    for (let i = 0; i < 10000; i++) last = mt.next();
    expect(last).toBe(4123659995);
  });

  it("keeps only the low 32 bits of the seed, like torch", () => {
    expect(new MT19937(2 ** 32 + 7).next()).toBe(new MT19937(7).next());
  });
});

describe("torchRandn", () => {
  // Values torch prints for these calls (known from the PyTorch docs and tutorials).
  it("matches torch.manual_seed(0); torch.randn(2, 3) (fewer than 16 values: double path)", () => {
    expect(round4(torchRandn(6, 0))).toEqual([1.541, -0.2934, -2.1788, 0.5684, -1.0845, -1.3986]);
  });

  it("matches torch.manual_seed(42); torch.randn(3)", () => {
    expect(round4(torchRandn(3, 42))).toEqual([0.3367, 0.1288, 0.2345]);
  });

  it("matches torch.manual_seed(0); torch.randn(4, 4) (16 values: float path)", () => {
    expect(round4(torchRandn(16, 0))).toEqual([
      -1.1258, -1.1524, -0.2506, -0.4339, 0.8487, 0.692, -0.316, -2.1152, 0.3223, -1.2633, 0.35, 0.3081, 0.1198, 1.2377, 1.1168,
      -0.2473,
    ]);
  });

  it("matches the fixture checked against real torch (incl. the redone last 16 and the Weeping Gate latent)", () => {
    for (const { n, seed, values } of fixture.cases) {
      const a = torchRandn(n, seed);
      for (const [i, v] of Object.entries(values)) expect(a[Number(i)], `n=${n} seed=${seed} [${i}]`).toBeCloseTo(v, 6);
    }
  });

  it("is roughly standard normal on a big latent", () => {
    const a = torchRandn(16 * 128 * 128, 54460995);
    const mean = a.reduce((s, v) => s + v, 0) / a.length;
    const std = Math.sqrt(a.reduce((s, v) => s + (v - mean) ** 2, 0) / a.length);
    expect(Math.abs(mean)).toBeLessThan(0.01);
    expect(Math.abs(std - 1)).toBeLessThan(0.01);
  });
});
