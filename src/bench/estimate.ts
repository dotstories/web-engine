/** Krea 2 Turbo sizes (from the model file) and a per-image time estimate from measured kernels. */
export const KREA = { blocks: 28, hidden: 6144, steps: 8, textTokens: 512, pixelsPerToken: 16 };

/** The matmuls of one block: y[tokens, n] = x[tokens, k] · Wᵀ, `count` times. */
export const BLOCK_MATMULS = [
  { name: "attention q, out, gate", n: 6144, k: 6144, count: 3 },
  { name: "attention k, v", n: 1536, k: 6144, count: 2 },
  { name: "mlp gate, up", n: 16384, k: 6144, count: 2 },
  { name: "mlp down", n: 6144, k: 16384, count: 1 },
];

export const blockParams = () => BLOCK_MATMULS.reduce((sum, s) => sum + s.n * s.k * s.count, 0);

export function tokensFor(width: number, height: number): number {
  const side = (px: number) => Math.ceil(px / KREA.pixelsPerToken);
  return side(width) * side(height) + KREA.textTokens;
}

export type Timing = { n: number; k: number; count: number; ms: number };

/** Attention isn't written yet: assume it runs as fast as the matmuls (QKᵀ and PV: 4·T²·hidden flops). */
export function estimateImage(timings: Timing[], tokens: number) {
  const matmulMs = timings.reduce((sum, t) => sum + t.count * t.ms, 0);
  const flops = timings.reduce((sum, t) => sum + t.count * 2 * tokens * t.n * t.k, 0);
  const tflops = flops / (matmulMs * 1e9);
  const attentionMs = (4 * tokens ** 2 * KREA.hidden) / (tflops * 1e9);
  const blockMs = matmulMs + attentionMs;
  return { tflops, matmulMs, blockMs, imageSeconds: (blockMs * KREA.blocks * KREA.steps) / 1000 };
}
