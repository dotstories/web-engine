import { Graph } from "../gpu/graph";
import { padTo4, softmaxWgsl, softmaxWorkgroups, transposeHeadsWgsl, transposeWorkgroups } from "./attention-parts";
import { matmulF32Wgsl, matmulParams, matmulWorkgroups } from "./matmul";

/**
 * `chunk`: queries per pass (default all). Smaller chunks bound the score memory at tokens × chunk.
 * `causal`: each token only attends to itself and earlier tokens (LLMs such as the text encoder).
 */
export type AttentionShape = { tokens: number; heads: number; kvHeads: number; dim: number; chunk?: number; causal?: boolean };

/**
 * Scaled dot-product attention with grouped-query heads, as in Krea 2 (48 query heads share
 * 12 key/value heads; query head h uses kv head ⌊h / (heads / kvHeads)⌋), no mask or causal.
 * q: [tokens, heads, dim], k, v: [tokens, kvHeads, dim] → out: [tokens, heads, dim].
 *
 * Built from the matmul kernel, one head at a time:
 *   S = Q_h · K_gᵀ / √dim  →  P = softmax(S)  →  O_h = P · V_g   (V transposed once).
 * S and P are [chunk, tokens] (rows padded to a multiple of 4) and reused for every head and chunk.
 */
export function attention(graph: Graph, q: GPUBuffer, k: GPUBuffer, v: GPUBuffer, shape: AttentionShape): GPUBuffer {
  const { tokens, heads, kvHeads, dim } = shape;
  const chunk = Math.min(shape.chunk ?? tokens, tokens);
  if (heads % kvHeads) throw new Error("heads must be a multiple of kvHeads");
  const stride = padTo4(tokens);
  const group = heads / kvHeads;

  const vt = graph.buffer(kvHeads * dim * stride * 4);
  graph.run(transposeHeadsWgsl, [graph.buffer(new Uint32Array([tokens, kvHeads, dim, stride])), v, vt], transposeWorkgroups(tokens, kvHeads, dim));

  const scores = graph.buffer(chunk * stride * 4);
  const probs = graph.buffer(chunk * stride * 4);
  const out = graph.buffer(tokens * heads * dim * 4);

  for (let h = 0; h < heads; h++) {
    const g = Math.floor(h / group);
    for (let q0 = 0; q0 < tokens; q0 += chunk) {
      const rows = Math.min(chunk, tokens - q0);
      const qk = { m: rows, n: tokens, k: dim };
      const qkLayout = { xStride: heads * dim, xOffset: q0 * heads * dim + h * dim, wStride: kvHeads * dim, wOffset: g * dim, yStride: stride };
      graph.run(matmulF32Wgsl, [graph.buffer(matmulParams(qk, 1 / Math.sqrt(dim), qkLayout)), q, k, scores], matmulWorkgroups(qk));

      graph.run(softmaxWgsl, [graph.buffer(new Uint32Array([rows, tokens, stride, shape.causal ? 1 : 0, q0])), scores, probs], softmaxWorkgroups(rows));

      const pv = { m: rows, n: dim, k: stride };
      const pvLayout = { xStride: stride, wStride: stride, wOffset: g * dim * stride, yStride: heads * dim, yOffset: q0 * heads * dim + h * dim };
      graph.run(matmulF32Wgsl, [graph.buffer(matmulParams(pv, 1, pvLayout)), probs, vt, out], matmulWorkgroups(pv));
    }
  }
  [vt, scores, probs].forEach((b) => graph.free(b)); // at 2 MP: 0.66 GB, reused by the next temporaries
  return out;
}
