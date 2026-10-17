import { Graph } from "../gpu/graph";
import { attention } from "../kernels/attention";
import { padTo4, softmaxWgsl, softmaxWorkgroups } from "../kernels/attention-parts";
import { matmulF32Wgsl, matmulParams, matmulWorkgroups } from "../kernels/matmul";

export type AttentionShape = { tokens: number; heads: number; kvHeads: number; dim: number };
export type AttentionTiming = { name: string; ms: number; tflops?: number };

/** Krea 2 at the Dark Fantasy Kit size: 1928×1088 → 121×68 = 8228 image tokens + 512 prompt tokens. */
export const KREA2_ATTENTION: AttentionShape = { tokens: 8228 + 512, heads: 48, kvHeads: 12, dim: 128 };

/**
 * Milliseconds for one block's attention (all heads) and for its parts, on zeroed buffers:
 * QK (scores, per head), softmax, PV (per head), and the whole attention() as the engine runs it.
 */
export async function timeAttention(device: GPUDevice, shape: AttentionShape = KREA2_ATTENTION): Promise<AttentionTiming[]> {
  const { tokens, heads, kvHeads, dim } = shape;
  const graph = new Graph(device);
  const stride = padTo4(tokens);
  const q = graph.buffer(tokens * heads * dim * 4);
  const k = graph.buffer(tokens * kvHeads * dim * 4);
  const v = graph.buffer(tokens * kvHeads * dim * 4);
  const vt = graph.buffer(kvHeads * dim * stride * 4);
  const scores = graph.buffer(tokens * stride * 4);
  const probs = graph.buffer(tokens * stride * 4);
  const out = graph.buffer(tokens * heads * dim * 4);
  const qk = { m: tokens, n: tokens, k: dim };
  const pv = { m: tokens, n: dim, k: stride };
  const flops = 2 * tokens * tokens * dim * heads; // QK or PV, all heads

  const parts: Record<string, () => void> = {
    "QK (scores)": () => {
      const params = graph.buffer(matmulParams(qk, 1 / Math.sqrt(dim), { xStride: heads * dim, wStride: kvHeads * dim, yStride: stride }));
      for (let h = 0; h < heads; h++) graph.run(matmulF32Wgsl, [params, q, k, scores], matmulWorkgroups(qk));
    },
    softmax: () => {
      const params = graph.buffer(new Uint32Array([tokens, tokens, stride, 0, 0]));
      for (let h = 0; h < heads; h++) graph.run(softmaxWgsl, [params, scores, probs], softmaxWorkgroups(tokens));
    },
    "PV (weighted values)": () => {
      const params = graph.buffer(matmulParams(pv, 1, { xStride: stride, wStride: stride, yStride: heads * dim }));
      for (let h = 0; h < heads; h++) graph.run(matmulF32Wgsl, [params, probs, vt, out], matmulWorkgroups(pv));
    },
    "attention (all of it)": () => void attention(graph, q, k, v, shape),
  };

  const time = async (queue: () => void) => {
    const mark = graph.mark();
    queue();
    graph.release(mark); // submits
    await device.queue.onSubmittedWorkDone();
  };

  try {
    const timings: AttentionTiming[] = [];
    for (const [name, queue] of Object.entries(parts)) {
      await time(queue); // warm up: pipelines, pool
      const start = performance.now();
      await time(queue);
      const ms = performance.now() - start;
      timings.push(name.startsWith("QK") || name.startsWith("PV") ? { name, ms, tflops: flops / (ms * 1e9) } : { name, ms });
    }
    return timings;
  } finally {
    graph.destroy();
  }
}
