import { Graph } from "../gpu/graph";
import { attention } from "../kernels/attention";
import { AXES, positions, ropeTables } from "../kernels/rope";
import { block, KREA_BLOCK, modulation, type BlockConfig, type BlockWeights } from "../model/block";
import { gatedAdd, linear, noMod, norm, rotate, sigmoidGate, swiglu, type Linear, type Rope } from "../model/ops";

export type BlockSize = { textTokens: number; rows: number; columns: number }; // patch grid
export type OpTiming = { name: string; kind: "linear" | "attention" | "other"; ms: number };

/** The Dark Fantasy Kit size: 1928×1088 → 68 × 121 patches, 512 prompt tokens. */
export const KIT_SIZE: BlockSize = { textTokens: 512, rows: 68, columns: 121 };

/**
 * Where one Krea 2 block's time goes: the same ops as block(), in the same order, each run on its own
 * (submit, wait), on zeroed fp8 weights (values don't change the speed). Plus the whole block() at once.
 */
export async function profileBlock(device: GPUDevice, size: BlockSize = KIT_SIZE, cfg: BlockConfig = KREA_BLOCK) {
  const { hidden, mlp, heads, kvHeads, headDim, eps } = cfg;
  const graph = new Graph(device);
  const t = size.textTokens + size.rows * size.columns;
  const fp8 = (n: number, k: number): Linear => ({ buffer: graph.buffer(n * k), dtype: "fp8", scale: 1, n, k });
  const w: BlockWeights = {
    wq: fp8(heads * headDim, hidden),
    wk: fp8(kvHeads * headDim, hidden),
    wv: fp8(kvHeads * headDim, hidden),
    gate: fp8(hidden, hidden),
    wo: fp8(hidden, heads * headDim),
    mlpGate: fp8(mlp, hidden),
    mlpUp: fp8(mlp, hidden),
    mlpDown: fp8(hidden, mlp),
    norm1: graph.buffer(hidden * 4),
    norm2: graph.buffer(hidden * 4),
    normQ: graph.buffer(headDim * 4),
    normK: graph.buffer(headDim * 4),
    table: graph.buffer(6 * hidden * 4),
  };
  const axes = headDim === 128 ? AXES : [headDim / 4, (3 * headDim) / 8, (3 * headDim) / 8]; // tiny test blocks
  const tables = ropeTables(positions(size.textTokens, size.rows, size.columns), axes);
  const rope: Rope = { cos: graph.buffer(tables.cos), sin: graph.buffer(tables.sin), pairs: tables.pairs, tokens: t };
  const x = graph.buffer(t * hidden * 4);
  const tvec = graph.buffer(6 * hidden * 4);
  await device.queue.onSubmittedWorkDone();

  const timings: OpTiming[] = [];
  const step = async <T>(name: string, kind: OpTiming["kind"], run: () => T): Promise<T> => {
    await device.queue.onSubmittedWorkDone();
    const start = performance.now();
    const out = run();
    graph.submit();
    await device.queue.onSubmittedWorkDone();
    timings.push({ name, kind, ms: performance.now() - start });
    return out;
  };

  try {
    // Warm up: compile every pipeline once.
    const warm = graph.mark();
    block(graph, w, cfg, x, tvec, rope);
    graph.release(warm);
    await device.queue.onSubmittedWorkDone();

    const mark = graph.mark();
    const mod = await step("modulation", "other", () => modulation(graph, tvec, w.table, hidden));
    const none = noMod(graph, headDim);
    const h = await step("norm 1", "other", () => norm(graph, x, w.norm1, t, hidden, { scale: mod.prescale, shift: mod.preshift }, eps));
    const q0 = await step("q", "linear", () => linear(graph, h, w.wq, t));
    const q1 = await step("q norm", "other", () => norm(graph, q0, w.normQ, t * heads, headDim, none, eps));
    const k0 = await step("k", "linear", () => linear(graph, h, w.wk, t));
    const k1 = await step("k norm", "other", () => norm(graph, k0, w.normK, t * kvHeads, headDim, none, eps));
    const q = await step("rope q", "other", () => rotate(graph, q1, heads, rope));
    const k = await step("rope k", "other", () => rotate(graph, k1, kvHeads, rope));
    const v = await step("v", "linear", () => linear(graph, h, w.wv, t));
    const o = await step("attention", "attention", () => attention(graph, q, k, v, { tokens: t, heads, kvHeads, dim: headDim }));
    const g = await step("gate", "linear", () => linear(graph, h, w.gate, t));
    const og = await step("sigmoid gate", "other", () => sigmoidGate(graph, o, g, t * hidden));
    const a = await step("out", "linear", () => linear(graph, og, w.wo, t));
    const x1 = await step("gated add 1", "other", () => gatedAdd(graph, x, a, mod.pregate, t * hidden, hidden));
    const h2 = await step("norm 2", "other", () => norm(graph, x1, w.norm2, t, hidden, { scale: mod.postscale, shift: mod.postshift }, eps));
    const mg = await step("mlp gate", "linear", () => linear(graph, h2, w.mlpGate, t));
    const mu = await step("mlp up", "linear", () => linear(graph, h2, w.mlpUp, t));
    const s = await step("swiglu", "other", () => swiglu(graph, mg, mu, t * mlp));
    const d = await step("mlp down", "linear", () => linear(graph, s, w.mlpDown, t));
    await step("gated add 2", "other", () => gatedAdd(graph, x1, d, mod.postgate, t * hidden, hidden));
    graph.release(mark);

    const whole = graph.mark();
    await step("whole block (one submit)", "other", () => block(graph, w, cfg, x, tvec, rope));
    graph.release(whole);
    const ops = timings.slice(0, -1);
    const sum = (kind: OpTiming["kind"]) => ops.filter((o) => o.kind === kind).reduce((total, o) => total + o.ms, 0);
    return { tokens: t, ops, block: timings.at(-1)!.ms, linear: sum("linear"), attention: sum("attention"), other: sum("other") };
  } finally {
    graph.destroy();
  }
}
