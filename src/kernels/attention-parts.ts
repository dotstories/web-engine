import { runCompute } from "../gpu/compute";

/**
 * Two small steps of attention:
 * - softmax per row over the first `cols` values of each row (rows are `stride` apart); the padding
 *   values [cols, stride) are written as 0 so a following matmul with K = stride stays exact.
 *   causal = 1: row r (query row0 + r) only sees columns 0..row0 + r, the rest becomes 0 (LLM attention).
 * - transposeHeads: V [tokens, heads, dim] → Vᵀ [heads, dim, stride] (tokens padded to `stride`),
 *   so P·V can run as a matmul with Vᵀ as the [N, K] weight.
 */
const THREADS = 256;

export const softmaxWgsl = /* wgsl */ `
struct Params { rows: u32, cols: u32, stride: u32, causal: u32, row0: u32 }
@group(0) @binding(0) var<storage, read> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

var<workgroup> partial: array<f32, ${THREADS}>;

fn reduce(v: f32, lid: u32, is_max: bool) -> f32 {
  partial[lid] = v;
  workgroupBarrier();
  for (var s = ${THREADS / 2}u; s > 0u; s >>= 1u) {
    if (lid < s) {
      partial[lid] = select(partial[lid] + partial[lid + s], max(partial[lid], partial[lid + s]), is_max);
    }
    workgroupBarrier();
  }
  let result = partial[0];
  workgroupBarrier();
  return result;
}

@compute @workgroup_size(${THREADS})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let row = wg.x + wg.y * 65535u;
  if (row >= p.rows) { return; }
  let base = row * p.stride;
  let cols = select(p.cols, min(p.cols, p.row0 + row + 1u), p.causal == 1u);

  var m = -3.4e38;
  for (var i = lid.x; i < cols; i += ${THREADS}u) { m = max(m, x[base + i]); }
  m = reduce(m, lid.x, true);

  var sum = 0.0;
  for (var i = lid.x; i < cols; i += ${THREADS}u) { sum += exp(x[base + i] - m); }
  sum = reduce(sum, lid.x, false);

  for (var i = lid.x; i < p.stride; i += ${THREADS}u) {
    out[base + i] = select(0.0, exp(x[base + i] - m) / sum, i < cols);
  }
}
`;

export const transposeHeadsWgsl = /* wgsl */ `
struct Params { tokens: u32, heads: u32, dim: u32, stride: u32 }
@group(0) @binding(0) var<storage, read> p: Params;
@group(0) @binding(1) var<storage, read> v: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(${THREADS})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let d = id.x;
  let t = id.y + id.z * 65535u;
  if (d >= p.heads * p.dim || t >= p.tokens) { return; }
  // d runs over (head, dim) of token t: reads are contiguous, writes jump by stride
  out[d * p.stride + t] = v[t * p.heads * p.dim + d];
}
`;

export const softmaxWorkgroups = (rows: number): [number, number] => [Math.min(rows, 65535), Math.ceil(rows / 65535)];

export const transposeWorkgroups = (tokens: number, heads: number, dim: number): [number, number, number] => [
  Math.ceil((heads * dim) / THREADS),
  Math.min(tokens, 65535),
  Math.ceil(tokens / 65535),
];

/** Rows of an attention score matrix are padded to a multiple of 4 for the matmuls. */
export const padTo4 = (n: number) => Math.ceil(n / 4) * 4;

/** One-shot runs (tests, debugging). */
export async function softmax(device: GPUDevice, x: Float32Array, rows: number, cols: number, stride = cols, causal = false) {
  const [out] = await runCompute(device, {
    code: softmaxWgsl,
    inputs: [new Uint32Array([rows, cols, stride, causal ? 1 : 0, 0]), x],
    outputs: [rows * stride * 4],
    workgroups: softmaxWorkgroups(rows),
  });
  return new Float32Array(out);
}

export async function transposeHeads(device: GPUDevice, v: Float32Array, tokens: number, heads: number, dim: number) {
  const stride = padTo4(tokens);
  const [out] = await runCompute(device, {
    code: transposeHeadsWgsl,
    inputs: [new Uint32Array([tokens, heads, dim, stride]), v],
    outputs: [heads * dim * stride * 4],
    workgroups: transposeWorkgroups(tokens, heads, dim),
  });
  return new Float32Array(out);
}
