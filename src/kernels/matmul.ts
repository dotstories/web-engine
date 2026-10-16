import { runCompute } from "../gpu/compute";
import { fp8Wgsl } from "./fp8";

/**
 * y[M, N] = x[M, K] · Wᵀ × scale, with W stored like PyTorch's nn.Linear: [N, K].
 * W is either fp8 e4m3fn packed 4 per u32 (straight from the model file) or f32 (e.g. K or Vᵀ in
 * attention). Accumulates in f32.
 *
 * Rows can be strided and offset (in values), so one attention head can be picked out of
 * [tokens, heads, 128] without copying: row r of x starts at xOffset + r · xStride, and so on.
 * Within a row, k is contiguous.
 *
 * Tiles: 128×128 outputs per workgroup (16×16 threads × 8×8 each), K in steps of 16.
 * x and W are read 4 values at a time; W is decoded once per tile into workgroup memory,
 * stored k-major so each thread reads 8 neighbours per step.
 *
 * The inner code is generated fully unrolled with named variables: arrays indexed in loops can
 * end up in slow memory instead of registers after WGSL is translated for the GPU driver.
 * Needs K, x/W strides and offsets to be multiples of 4 (true for every Krea 2 layer).
 */
export const TILE = 128;

export type Weights = "fp8" | "f32";

const range = (n: number) => Array.from({ length: n }, (_, i) => i);
const lines = (n: number, line: (i: number) => string) => range(n).map(line).join("\n");

const loadW: Record<Weights, string> = {
  fp8: /* wgsl */ `
    var word = 0u; // decodes to 0.0
    if (col0 + r < p.n && kq < k4) { word = w[(p.w_offset + (col0 + r) * p.w_stride) / 4u + kq]; }
    ws[at] = e4m3_to_f32(fp8_byte(word, 0u));
    ws[at + TN] = e4m3_to_f32(fp8_byte(word, 1u));
    ws[at + 2u * TN] = e4m3_to_f32(fp8_byte(word, 2u));
    ws[at + 3u * TN] = e4m3_to_f32(fp8_byte(word, 3u));`,
  f32: /* wgsl */ `
    var u = vec4<f32>(0.0);
    if (col0 + r < p.n && kq < k4) { u = w[(p.w_offset + (col0 + r) * p.w_stride) / 4u + kq]; }
    ws[at] = u.x;
    ws[at + TN] = u.y;
    ws[at + 2u * TN] = u.z;
    ws[at + 3u * TN] = u.w;`,
};

const load = (t: number, weights: Weights) => /* wgsl */ `
  {
    let e = li + ${t * 256}u;
    let r = e / 4u;
    let q = e % 4u;
    let kq = k0 / 4u + q;
    let at = 4u * q * TM + r;
    var v = vec4<f32>(0.0);
    if (row0 + r < p.m && kq < k4) { v = x[(p.x_offset + (row0 + r) * p.x_stride) / 4u + kq]; }
    xs[at] = v.x;
    xs[at + TM] = v.y;
    xs[at + 2u * TM] = v.z;
    xs[at + 3u * TM] = v.w;
${loadW[weights]}
  }`;

export const matmulWgsl = (weights: Weights) => /* wgsl */ `
${weights === "fp8" ? fp8Wgsl : ""}
struct Params {
  m: u32, n: u32, k: u32, scale: f32,
  x_stride: u32, w_stride: u32, y_stride: u32,
  x_offset: u32, w_offset: u32, y_offset: u32,
}

@group(0) @binding(0) var<storage, read> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> w: array<${weights === "fp8" ? "u32" : "vec4<f32>"}>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

const TM = 128u;
const TN = 128u;
const TK = 16u;
var<workgroup> xs: array<f32, 2048>; // [TK][TM]
var<workgroup> ws: array<f32, 2048>; // [TK][TN]

fn store4(r: u32, c: u32, v: vec4<f32>) {
  let o = p.y_offset + r * p.y_stride + c;
  if (c < p.n) { y[o] = v.x * p.scale; }
  if (c + 1u < p.n) { y[o + 1u] = v.y * p.scale; }
  if (c + 2u < p.n) { y[o + 2u] = v.z * p.scale; }
  if (c + 3u < p.n) { y[o + 3u] = v.w * p.scale; }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u,
        @builtin(local_invocation_index) li: u32) {
  let row0 = wg.y * TM;
  let col0 = wg.x * TN;
  let k4 = p.k / 4u;
${lines(16, (i) => `  var acc${i} = vec4<f32>(0.0);`)}

  for (var k0 = 0u; k0 < p.k; k0 += TK) {
${load(0, weights)}
${load(1, weights)}
    workgroupBarrier();

    for (var kk = 0u; kk < TK; kk++) {
      let xa = kk * TM + lid.y * 8u;
      let wa = kk * TN + lid.x * 8u;
${lines(8, (i) => `      let a${i} = xs[xa + ${i}u];`)}
      let b0 = vec4<f32>(ws[wa], ws[wa + 1u], ws[wa + 2u], ws[wa + 3u]);
      let b1 = vec4<f32>(ws[wa + 4u], ws[wa + 5u], ws[wa + 6u], ws[wa + 7u]);
${lines(8, (i) => `      acc${2 * i} += a${i} * b0;\n      acc${2 * i + 1} += a${i} * b1;`)}
    }
    workgroupBarrier();
  }

  let c0 = col0 + lid.x * 8u;
${lines(8, (i) => `  if (row0 + lid.y * 8u + ${i}u < p.m) { store4(row0 + lid.y * 8u + ${i}u, c0, acc${2 * i}); store4(row0 + lid.y * 8u + ${i}u, c0 + 4u, acc${2 * i + 1}); }`)}
}
`;

export const matmulFp8Wgsl = matmulWgsl("fp8");
export const matmulF32Wgsl = matmulWgsl("f32");

export type MatmulShape = { m: number; n: number; k: number };
/** Row strides and offsets in values. Defaults: contiguous x [M, K], W [N, K], y [M, N]. */
export type MatmulLayout = { xStride?: number; wStride?: number; yStride?: number; xOffset?: number; wOffset?: number; yOffset?: number };

/** Params buffer for matmulWgsl. */
export function matmulParams({ m, n, k }: MatmulShape, scale: number, layout: MatmulLayout = {}): Uint32Array<ArrayBuffer> {
  const { xStride = k, wStride = k, yStride = n, xOffset = 0, wOffset = 0, yOffset = 0 } = layout;
  for (const [name, v] of Object.entries({ K: k, xStride, wStride, xOffset, wOffset })) {
    if (v % 4) throw new Error(`${name} must be a multiple of 4, got ${v}`);
  }
  const params = new Uint32Array(12);
  params.set([m, n, k, 0, xStride, wStride, yStride, xOffset, wOffset, yOffset]);
  new Float32Array(params.buffer)[3] = scale;
  return params;
}

export const matmulWorkgroups = ({ m, n }: MatmulShape): [number, number] => [Math.ceil(n / TILE), Math.ceil(m / TILE)];

const padTo4 = (w: Uint8Array) => {
  const packed = new Uint8Array(Math.ceil(w.length / 4) * 4);
  packed.set(w);
  return new Uint32Array(packed.buffer);
};

/** One-shot run (tests, debugging). The engine keeps weights on the GPU instead. */
export async function matmulFp8(device: GPUDevice, x: Float32Array, w: Uint8Array, scale: number, shape: MatmulShape) {
  const { m, n, k } = shape;
  if (x.length !== m * k || w.length !== n * k) throw new Error("x or w does not match the shape");
  const [y] = await runCompute(device, {
    code: matmulFp8Wgsl,
    inputs: [matmulParams(shape, scale), x, padTo4(w)],
    outputs: [m * n * 4],
    workgroups: matmulWorkgroups(shape),
  });
  return new Float32Array(y);
}

/** One-shot f32 run with any layout; `outputLength` is the size of y in values. */
export async function matmulF32(
  device: GPUDevice,
  x: Float32Array,
  w: Float32Array,
  scale: number,
  shape: MatmulShape,
  layout: MatmulLayout = {},
  outputLength = shape.m * shape.n,
) {
  const [y] = await runCompute(device, {
    code: matmulF32Wgsl,
    inputs: [matmulParams(shape, scale, layout), x, w],
    outputs: [outputLength * 4],
    workgroups: matmulWorkgroups(shape),
  });
  return new Float32Array(y);
}
