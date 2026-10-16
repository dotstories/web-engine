import { runCompute } from "../gpu/compute";

/**
 * RMSNorm per row, with Krea 2's modulation:
 *   out = x / rms(x) · (offset + weight) · (1 + scale) + shift,   rms(x) = sqrt(mean(x²) + eps)
 * offset = 1 for Krea 2's zero-centered norms (weights stored as "1 + w"), 0 for plain RMSNorm.
 * scale/shift are the block's modulation vectors (zeros when there is none, e.g. q/k norm).
 * One workgroup per row; rows beyond 65535 use the y dimension.
 */
export const ROWS_PER_DIM = 65535;
const THREADS = 128;

export const rmsnormWgsl = /* wgsl */ `
struct Params { rows: u32, d: u32, eps: f32, offset: f32 }

@group(0) @binding(0) var<storage, read> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> scale: array<f32>;
@group(0) @binding(4) var<storage, read> shift: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

var<workgroup> partial: array<f32, ${THREADS}>;

@compute @workgroup_size(${THREADS})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let row = wg.x + wg.y * ${ROWS_PER_DIM}u;
  if (row >= p.rows) { return; }
  let base = row * p.d;

  var sum = 0.0;
  for (var i = lid.x; i < p.d; i += ${THREADS}u) {
    let v = x[base + i];
    sum += v * v;
  }
  partial[lid.x] = sum;
  workgroupBarrier();
  for (var s = ${THREADS / 2}u; s > 0u; s >>= 1u) {
    if (lid.x < s) { partial[lid.x] += partial[lid.x + s]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(partial[0] / f32(p.d) + p.eps);

  for (var i = lid.x; i < p.d; i += ${THREADS}u) {
    out[base + i] = x[base + i] * inv * (p.offset + weight[i]) * (1.0 + scale[i]) + shift[i];
  }
}
`;

export type RmsNormOptions = { eps?: number; offset?: number; scale?: Float32Array; shift?: Float32Array };

export function rmsnormParams(rows: number, d: number, eps: number, offset: number): Uint32Array<ArrayBuffer> {
  const params = new Uint32Array(4);
  params.set([rows, d]);
  new Float32Array(params.buffer).set([eps, offset], 2);
  return params;
}

export const rmsnormWorkgroups = (rows: number): [number, number] => [Math.min(rows, ROWS_PER_DIM), Math.ceil(rows / ROWS_PER_DIM)];

/** One-shot run (tests, debugging). */
export async function rmsnorm(device: GPUDevice, x: Float32Array, weight: Float32Array, options: RmsNormOptions = {}) {
  const d = weight.length;
  const rows = x.length / d;
  if (!Number.isInteger(rows)) throw new Error("x is not a whole number of rows");
  const { eps = 1e-5, offset = 1, scale = new Float32Array(d), shift = new Float32Array(d) } = options;
  const [out] = await runCompute(device, {
    code: rmsnormWgsl,
    inputs: [rmsnormParams(rows, d, eps, offset), x, weight, scale, shift],
    outputs: [x.byteLength],
    workgroups: rmsnormWorkgroups(rows),
  });
  return new Float32Array(out);
}
