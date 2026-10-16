import { runCompute } from "../gpu/compute";

/**
 * Kernels for Krea 2's text fusion (12 text-encoder layers per token → one vector per token):
 * - shortAttention: attention inside many short sequences at once (each token's 12 layers),
 *   q, k, v, out: [batch · seq, heads, dim], same number of q and kv heads, no mask, no RoPE.
 *   One workgroup per (sequence, head); seq ≤ 32.
 * - layerSum: out[t, d] = Σ_l w[l] · x[t, l, d] (the 12 → 1 projector).
 */
export const MAX_SEQ = 32;
const THREADS = 128;

export const shortAttentionWgsl = /* wgsl */ `
struct Params { batch: u32, seq: u32, heads: u32, dim: u32, scale: f32 }
@group(0) @binding(0) var<storage, read> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

var<workgroup> scores: array<f32, ${MAX_SEQ * MAX_SEQ}>;

fn at(b: u32, i: u32, h: u32, d: u32) -> u32 {
  return ((b * p.seq + i) * p.heads + h) * p.dim + d;
}

@compute @workgroup_size(${THREADS})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let b = wg.x + wg.z * 65535u;
  let h = wg.y;
  if (b >= p.batch) { return; }
  let s = p.seq;

  for (var e = lid.x; e < s * s; e += ${THREADS}u) {
    let i = e / s;
    let j = e % s;
    var dot = 0.0;
    for (var d = 0u; d < p.dim; d++) { dot += q[at(b, i, h, d)] * k[at(b, j, h, d)]; }
    scores[e] = dot * p.scale;
  }
  workgroupBarrier();

  if (lid.x < s) {
    let row = lid.x * s;
    var m = -3.4e38;
    for (var j = 0u; j < s; j++) { m = max(m, scores[row + j]); }
    var sum = 0.0;
    for (var j = 0u; j < s; j++) {
      let e = exp(scores[row + j] - m);
      scores[row + j] = e;
      sum += e;
    }
    for (var j = 0u; j < s; j++) { scores[row + j] /= sum; }
  }
  workgroupBarrier();

  for (var e = lid.x; e < s * p.dim; e += ${THREADS}u) {
    let i = e / p.dim;
    let d = e % p.dim;
    var o = 0.0;
    for (var j = 0u; j < s; j++) { o += scores[i * s + j] * v[at(b, j, h, d)]; }
    out[at(b, i, h, d)] = o;
  }
}
`;

export const layerSumWgsl = /* wgsl */ `
struct Params { tokens: u32, layers: u32, dim: u32 }
@group(0) @binding(0) var<storage, read> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x + id.y * 256u * 65535u;
  if (i >= p.tokens * p.dim) { return; }
  let t = i / p.dim;
  let d = i % p.dim;
  var sum = 0.0;
  for (var l = 0u; l < p.layers; l++) { sum += w[l] * x[(t * p.layers + l) * p.dim + d]; }
  out[i] = sum;
}
`;

export type ShortAttentionShape = { batch: number; seq: number; heads: number; dim: number };

export function shortAttentionParams({ batch, seq, heads, dim }: ShortAttentionShape): Uint32Array<ArrayBuffer> {
  if (seq > MAX_SEQ) throw new Error(`shortAttention handles sequences up to ${MAX_SEQ}, got ${seq}`);
  const params = new Uint32Array(5);
  params.set([batch, seq, heads, dim]);
  new Float32Array(params.buffer)[4] = 1 / Math.sqrt(dim);
  return params;
}

export const shortAttentionWorkgroups = ({ batch, heads }: ShortAttentionShape): [number, number, number] => [
  Math.min(batch, 65535),
  heads,
  Math.ceil(batch / 65535),
];

export const layerSumWorkgroups = (tokens: number, dim: number): [number, number] => {
  const groups = Math.ceil((tokens * dim) / 256);
  return [Math.min(groups, 65535), Math.ceil(groups / 65535)];
};

/** One-shot runs (tests, debugging). */
export async function shortAttention(device: GPUDevice, q: Float32Array, k: Float32Array, v: Float32Array, shape: ShortAttentionShape) {
  const [out] = await runCompute(device, {
    code: shortAttentionWgsl,
    inputs: [shortAttentionParams(shape), q, k, v],
    outputs: [q.byteLength],
    workgroups: shortAttentionWorkgroups(shape),
  });
  return new Float32Array(out);
}

export async function layerSum(device: GPUDevice, x: Float32Array, w: Float32Array, tokens: number, dim: number) {
  const [out] = await runCompute(device, {
    code: layerSumWgsl,
    inputs: [new Uint32Array([tokens, w.length, dim]), x, w],
    outputs: [tokens * dim * 4],
    workgroups: layerSumWorkgroups(tokens, dim),
  });
  return new Float32Array(out);
}
