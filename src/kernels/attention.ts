import { Graph } from "../gpu/graph";
import { padTo4, softmaxWgsl, softmaxWorkgroups, transposeHeadsWgsl, transposeWorkgroups } from "./attention-parts";
import { matmulF32Wgsl, matmulParams, matmulWorkgroups } from "./matmul";

export type AttentionShape = { tokens: number; heads: number; kvHeads: number; dim: number };

/**
 * Scaled dot-product attention with grouped-query heads, as in Krea 2 (48 query heads share
 * 12 key/value heads; query head h uses kv head ⌊h / (heads / kvHeads)⌋), no mask.
 * q: [tokens, heads, dim], k, v: [tokens, kvHeads, dim] → out: [tokens, heads, dim].
 *
 * Built from the matmul kernel, one head at a time:
 *   S = Q_h · K_gᵀ / √dim  →  P = softmax(S)  →  O_h = P · V_g   (V transposed once).
 * S and P are [tokens, tokens] (rows padded to a multiple of 4) and reused for every head.
 */
export function attention(graph: Graph, q: GPUBuffer, k: GPUBuffer, v: GPUBuffer, shape: AttentionShape): GPUBuffer {
  const { tokens, heads, kvHeads, dim } = shape;
  if (heads % kvHeads) throw new Error("heads must be a multiple of kvHeads");
  const stride = padTo4(tokens);
  const group = heads / kvHeads;

  const vt = graph.buffer(kvHeads * dim * stride * 4);
  graph.run(transposeHeadsWgsl, [graph.buffer(new Uint32Array([tokens, kvHeads, dim, stride])), v, vt], transposeWorkgroups(tokens, kvHeads, dim));

  const scores = graph.buffer(tokens * stride * 4);
  const probs = graph.buffer(tokens * stride * 4);
  const out = graph.buffer(tokens * heads * dim * 4);
  const softmaxParams = graph.buffer(new Uint32Array([tokens, tokens, stride]));

  for (let h = 0; h < heads; h++) {
    const g = Math.floor(h / group);
    const qk = { m: tokens, n: tokens, k: dim };
    const qkLayout = { xStride: heads * dim, xOffset: h * dim, wStride: kvHeads * dim, wOffset: g * dim, yStride: stride };
    graph.run(matmulF32Wgsl, [graph.buffer(matmulParams(qk, 1 / Math.sqrt(dim), qkLayout)), q, k, scores], matmulWorkgroups(qk));

    graph.run(softmaxWgsl, [softmaxParams, scores, probs], softmaxWorkgroups(tokens));

    const pv = { m: tokens, n: dim, k: stride };
    const pvLayout = { xStride: stride, wStride: stride, wOffset: g * dim * stride, yStride: heads * dim, yOffset: h * dim };
    graph.run(matmulF32Wgsl, [graph.buffer(matmulParams(pv, 1, pvLayout)), probs, vt, out], matmulWorkgroups(pv));
  }
  return out;
}
