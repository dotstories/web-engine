import { expect, it } from "vitest";
import { getDevice } from "../../src/gpu/compute";
import { Graph } from "../../src/gpu/graph";
import { positions, ropeTables } from "../../src/kernels/rope";
import { block, type BlockConfig, type BlockWeights } from "../../src/model/block";
import { compare } from "../../src/model/compare";
import type { Linear } from "../../src/model/ops";
import { e4m3 } from "../../src/weights/fp8";

// A tiny block with the same structure as Krea 2's (GQA 2:1, 3-axis RoPE over a 16-dim head).
const cfg: BlockConfig = { hidden: 64, mlp: 96, heads: 4, kvHeads: 2, headDim: 16, eps: 1e-5 };
const axes = [4, 6, 6];

let seed = 1;
const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32) * 2 - 1;
const floats = (n: number, size: number) => Float32Array.from({ length: n }, () => rand() * size);
const fp8 = (n: number) =>
  Uint8Array.from({ length: n }, () => {
    const b = Math.floor((rand() + 1) * 128) & 0xff;
    return (b & 0x7f) === 0x7f ? b - 1 : b; // no NaN codes
  });

type CpuLinear = { bytes: Uint8Array; scale: number; n: number; k: number };

// ---------- float64 reference, written from the formulas (not from the shaders) ----------
const mm = (x: Float64Array, w: CpuLinear, m: number) => {
  const y = new Float64Array(m * w.n);
  for (let i = 0; i < m; i++)
    for (let j = 0; j < w.n; j++) {
      let s = 0;
      for (let t = 0; t < w.k; t++) s += x[i * w.k + t] * e4m3(w.bytes[j * w.k + t]);
      y[i * w.n + j] = s * w.scale;
    }
  return y;
};
const rmsNorm = (x: Float64Array, w: Float32Array, rows: number, d: number, scale?: Float64Array, shift?: Float64Array) => {
  const y = new Float64Array(rows * d);
  for (let r = 0; r < rows; r++) {
    let ss = 0;
    for (let i = 0; i < d; i++) ss += x[r * d + i] ** 2;
    const inv = 1 / Math.sqrt(ss / d + cfg.eps);
    for (let i = 0; i < d; i++) y[r * d + i] = x[r * d + i] * inv * (1 + w[i]) * (1 + (scale?.[i] ?? 0)) + (shift?.[i] ?? 0);
  }
  return y;
};
const ropeCpu = (x: Float64Array, tokens: number, heads: number, cos: Float32Array, sin: Float32Array) => {
  const y = new Float64Array(x.length);
  const pairs = cfg.headDim / 2;
  for (let t = 0; t < tokens; t++)
    for (let h = 0; h < heads; h++)
      for (let j = 0; j < pairs; j++) {
        const i = ((t * heads + h) * pairs + j) * 2;
        const [c, s] = [cos[t * pairs + j], sin[t * pairs + j]];
        y[i] = x[i] * c - x[i + 1] * s;
        y[i + 1] = x[i + 1] * c + x[i] * s;
      }
  return y;
};

function reference(x: Float64Array, w: Record<string, CpuLinear>, vecs: Record<string, Float32Array>, temb: Float32Array, tokens: number, cos: Float32Array, sin: Float32Array) {
  const { hidden, heads, kvHeads, headDim } = cfg;
  const mod = Array.from({ length: 6 }, (_, i) => Float64Array.from({ length: hidden }, (_, j) => temb[i * hidden + j] + vecs.table[i * hidden + j]));
  const [prescale, preshift, pregate, postscale, postshift, postgate] = mod;

  const h = rmsNorm(x, vecs.norm1, tokens, hidden, prescale, preshift);
  const q = ropeCpu(rmsNorm(mm(h, w.wq, tokens), vecs.normQ, tokens * heads, headDim), tokens, heads, cos, sin);
  const k = ropeCpu(rmsNorm(mm(h, w.wk, tokens), vecs.normK, tokens * kvHeads, headDim), tokens, kvHeads, cos, sin);
  const v = mm(h, w.wv, tokens);
  const g = mm(h, w.gate, tokens);

  const o = new Float64Array(tokens * hidden);
  for (let hd = 0; hd < heads; hd++) {
    const kv = Math.floor(hd / (heads / kvHeads));
    for (let i = 0; i < tokens; i++) {
      const s = Array.from({ length: tokens }, (_, j) => {
        let dot = 0;
        for (let d = 0; d < headDim; d++) dot += q[(i * heads + hd) * headDim + d] * k[(j * kvHeads + kv) * headDim + d];
        return dot / Math.sqrt(headDim);
      });
      const mx = Math.max(...s);
      const e = s.map((z) => Math.exp(z - mx));
      const sum = e.reduce((a, b) => a + b, 0);
      for (let d = 0; d < headDim; d++) {
        let acc = 0;
        for (let j = 0; j < tokens; j++) acc += (e[j] / sum) * v[(j * kvHeads + kv) * headDim + d];
        o[(i * heads + hd) * headDim + d] = acc;
      }
    }
  }
  const gated = o.map((val, i) => val / (1 + Math.exp(-g[i])));
  const a = mm(gated, w.wo, tokens);
  const x1 = x.map((val, i) => val + pregate[i % hidden] * a[i]);

  const h2 = rmsNorm(x1, vecs.norm2, tokens, hidden, postscale, postshift);
  const gg = mm(h2, w.mlpGate, tokens);
  const uu = mm(h2, w.mlpUp, tokens);
  const m = gg.map((val, i) => (val / (1 + Math.exp(-val))) * uu[i]);
  const d = mm(m, w.mlpDown, tokens);
  return x1.map((val, i) => val + postgate[i % hidden] * d[i]);
}

it("runs a whole block on the GPU like the float64 reference", async () => {
  const { hidden, mlp, heads, kvHeads, headDim } = cfg;
  const shapes: Record<string, [number, number]> = {
    wq: [heads * headDim, hidden], wk: [kvHeads * headDim, hidden], wv: [kvHeads * headDim, hidden],
    gate: [hidden, hidden], wo: [hidden, hidden], mlpGate: [mlp, hidden], mlpUp: [mlp, hidden], mlpDown: [hidden, mlp],
  };
  const cpuW = Object.fromEntries(Object.entries(shapes).map(([name, [n, k]]) => [name, { bytes: fp8(n * k), scale: 0.002, n, k }]));
  const vecs = { norm1: floats(hidden, 0.5), norm2: floats(hidden, 0.5), normQ: floats(headDim, 0.5), normK: floats(headDim, 0.5), table: floats(6 * hidden, 0.3) };
  const temb = floats(6 * hidden, 0.3);
  const pos = positions(3, 2, 3);
  const tokens = pos.length;
  const tables = ropeTables(pos, axes);
  const x = floats(tokens * hidden, 2);

  const graph = new Graph(await getDevice());
  try {
    const lin = (c: CpuLinear): Linear => ({ buffer: graph.buffer(new Uint32Array(c.bytes.buffer)), scale: c.scale, n: c.n, k: c.k });
    const weights: BlockWeights = {
      wq: lin(cpuW.wq), wk: lin(cpuW.wk), wv: lin(cpuW.wv), gate: lin(cpuW.gate), wo: lin(cpuW.wo),
      mlpGate: lin(cpuW.mlpGate), mlpUp: lin(cpuW.mlpUp), mlpDown: lin(cpuW.mlpDown),
      norm1: graph.buffer(vecs.norm1), norm2: graph.buffer(vecs.norm2), normQ: graph.buffer(vecs.normQ), normK: graph.buffer(vecs.normK),
      table: vecs.table,
    };
    const rope = { cos: graph.buffer(tables.cos), sin: graph.buffer(tables.sin), pairs: tables.pairs, tokens };
    const out = block(graph, weights, cfg, graph.buffer(x), temb, rope);
    const gpu = new Float32Array(await graph.read(out, tokens * hidden * 4));

    const cpu = reference(Float64Array.from(x), cpuW, vecs, temb, tokens, tables.cos, tables.sin);
    const r = compare(gpu, cpu);
    expect(r.cosine).toBeGreaterThan(0.99999);
    expect(r.relativeError).toBeLessThan(1e-4);
  } finally {
    graph.destroy();
  }
});
