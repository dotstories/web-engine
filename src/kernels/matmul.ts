import { runCompute } from "../gpu/compute";
import { fp8Wgsl } from "./fp8";

/**
 * y[M, N] = x[M, K] · Wᵀ × scale, with W stored like PyTorch's nn.Linear: [N, K].
 * W is fp8 e4m3fn packed 4 per u32 or bf16 packed 2 per u32 (both straight from the model file),
 * or f32 (e.g. K or Vᵀ in attention). Accumulates in f32.
 *
 * Rows can be strided and offset (in values), so one attention head can be picked out of
 * [tokens, heads, 128] without copying: row r of x starts at xOffset + r · xStride, and so on.
 * Within a row, k is contiguous.
 *
 * Tiles: 128×128 outputs per workgroup (16×16 threads × 8×8 each), K in steps of 16.
 * x and W are read 4 values at a time; W is decoded once per tile into workgroup memory,
 * stored k-major so each thread reads 8 neighbours per step.
 *
 * The inner code is generated fully unrolled (also over the 16 k of a step) with named variables:
 * arrays indexed in loops can end up in slow memory instead of registers after WGSL is translated for
 * the GPU driver. Multiply-adds use fma() (see step()).
 * Needs K, x/W strides and offsets to be multiples of 4 (true for every Krea 2 layer).
 *
 * Input "conv3x3" turns it into a 3×3 convolution (padding 1) without an im2col copy: x is a
 * channel-last image [H·W, Cin], row m is a pixel, and K = 9 · Cin runs over (tap, channel), so the
 * loader gathers each pixel's 9 neighbours (zeros outside the image). W is [Cout, (ky, kx, Cin)].
 *
 * splitK: workgroup z sums only its slice of K (kSplit slices) into partial z of y [kSplit, M, yStride].
 * For narrow outputs with a long K (a LoRA's x · downᵀ: N = rank), where 128×128 tiles alone
 * would leave most of the GPU idle. Sum the partials afterwards (layerSum).
 *
 * accumulate: y += x · Wᵀ × scale instead of y = …, so a LoRA's update goes straight into the layer's
 * output (no extra buffer, no add pass).
 */
export const TILE = 128;

export type Weights = "fp8" | "bf16" | "f32";

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
  bf16: /* wgsl */ `
    var lo = 0u; // two bf16 per word, first one in the low half; 0 decodes to 0.0
    var hi = 0u;
    if (col0 + r < p.n && kq < k4) {
      let word = (p.w_offset + (col0 + r) * p.w_stride) / 2u + kq * 2u;
      lo = w[word];
      hi = w[word + 1u];
    }
    ws[at] = bitcast<f32>(lo << 16u);
    ws[at + TN] = bitcast<f32>(lo & 0xffff0000u);
    ws[at + 2u * TN] = bitcast<f32>(hi << 16u);
    ws[at + 3u * TN] = bitcast<f32>(hi & 0xffff0000u);`,
  f32: /* wgsl */ `
    var u = vec4<f32>(0.0);
    if (col0 + r < p.n && kq < k4) { u = w[(p.w_offset + (col0 + r) * p.w_stride) / 4u + kq]; }
    ws[at] = u.x;
    ws[at + TN] = u.y;
    ws[at + 2u * TN] = u.z;
    ws[at + 3u * TN] = u.w;`,
};

export type Input = "rows" | "conv3x3";

const loadX: Record<Input, string> = {
  rows: /* wgsl */ `
    if (row0 + r < p.m && kq < k4) { v = x[(p.x_offset + (row0 + r) * p.x_stride) / 4u + kq]; }`,
  conv3x3: /* wgsl */ `
    if (row0 + r < p.m && kq < k4) {
      let tap = kq * 4u / p.cin; // 0..8, then the channel inside the tap
      let c = kq * 4u % p.cin;
      let px = i32((row0 + r) % p.width) + i32(tap % 3u) - 1;
      let py = i32((row0 + r) / p.width) + i32(tap / 3u) - 1;
      if (px >= 0 && py >= 0 && px < i32(p.width) && py < i32(p.m / p.width)) {
        v = x[((u32(py) * p.width + u32(px)) * p.cin + c) / 4u];
      }
    }`,
};

const load = (t: number, weights: Weights, input: Input) => /* wgsl */ `
  {
    let e = li + ${t * 256}u;
    let r = e / 4u;
    let q = e % 4u;
    let kq = k0 / 4u + q;
    let at = 4u * q * TM + r;
    var v = vec4<f32>(0.0);${loadX[input]}
    xs[at] = v.x;
    xs[at + TM] = v.y;
    xs[at + 2u * TM] = v.z;
    xs[at + 3u * TM] = v.w;
${loadW[weights]}
  }`;

/**
 * One k of the inner loop, unrolled (kk is a constant). Explicit fma(): on D3D12 (Windows Chrome/Edge)
 * "acc += a * b" became a separate multiply and add, which capped the matmul at ~17 TFLOPS on an RTX 4080;
 * with fma() it reaches ~25 (steady f32 peak ~37.5).
 */
const step = (kk: number) => /* wgsl */ `
    {
      let xa = ${kk}u * TM + lid.y * 8u;
      let wa = ${kk}u * TN + lid.x * 8u;
${lines(8, (i) => `      let a${i} = vec4<f32>(xs[xa + ${i}u]);`)}
      let b0 = vec4<f32>(ws[wa], ws[wa + 1u], ws[wa + 2u], ws[wa + 3u]);
      let b1 = vec4<f32>(ws[wa + 4u], ws[wa + 5u], ws[wa + 6u], ws[wa + 7u]);
${lines(8, (i) => `      acc${2 * i} = fma(a${i}, b0, acc${2 * i});\n      acc${2 * i + 1} = fma(a${i}, b1, acc${2 * i + 1});`)}
    }`;

export type MatmulMode = { splitK?: boolean; accumulate?: boolean };

export const matmulWgsl = (weights: Weights, input: Input = "rows", { splitK = false, accumulate = false }: MatmulMode = {}) => /* wgsl */ `
${weights === "fp8" ? fp8Wgsl : ""}
struct Params {
  m: u32, n: u32, k: u32, scale: f32,
  x_stride: u32, w_stride: u32, y_stride: u32,
  x_offset: u32, w_offset: u32, y_offset: u32,
  width: u32, cin: u32, // conv3x3 only
  k_split: u32, // splitK only
}

@group(0) @binding(0) var<storage, read> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> w: array<${weights === "f32" ? "vec4<f32>" : "u32"}>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

const TM = 128u;
const TN = 128u;
const TK = 16u;
var<workgroup> xs: array<f32, 2048>; // [TK][TM]
var<workgroup> ws: array<f32, 2048>; // [TK][TN]
${splitK ? "var<private> z_rows = 0u; // rows before this workgroup's partial" : ""}

fn store4(r: u32, c: u32, v: vec4<f32>) {
  let o = p.y_offset + ${splitK ? "(z_rows + r)" : "r"} * p.y_stride + c;
${lines(4, (i) => {
  const at = i ? `o + ${i}u` : "o";
  const c = i ? `c + ${i}u` : "c";
  return `  if (${c} < p.n) { y[${at}] = ${accumulate ? `y[${at}] + ` : ""}v.${"xyzw"[i]} * p.scale; }`;
})}
}

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u,
        @builtin(local_invocation_index) li: u32) {
  let row0 = wg.y * TM;
  let col0 = wg.x * TN;
${
  splitK
    ? `  let kc = ((p.k + p.k_split - 1u) / p.k_split + TK - 1u) / TK * TK;
  let k_begin = wg.z * kc;
  let k_end = min(p.k, k_begin + kc);
  z_rows = wg.z * p.m;`
    : `  let k_begin = 0u;
  let k_end = p.k;`
}
  let k4 = k_end / 4u;
${lines(16, (i) => `  var acc${i} = vec4<f32>(0.0);`)}

  for (var k0 = k_begin; k0 < k_end; k0 += TK) {
${load(0, weights, input)}
${load(1, weights, input)}
    workgroupBarrier();

${lines(16, (kk) => step(kk))}
    workgroupBarrier();
  }

  let c0 = col0 + lid.x * 8u;
${lines(8, (i) => `  if (row0 + lid.y * 8u + ${i}u < p.m) { store4(row0 + lid.y * 8u + ${i}u, c0, acc${2 * i}); store4(row0 + lid.y * 8u + ${i}u, c0 + 4u, acc${2 * i + 1}); }`)}
}
`;

export const matmulFp8Wgsl = matmulWgsl("fp8");
export const matmulBf16Wgsl = matmulWgsl("bf16");
export const matmulF32Wgsl = matmulWgsl("f32");
export const conv3x3Wgsl = matmulWgsl("f32", "conv3x3");

export type MatmulShape = { m: number; n: number; k: number };
/** Row strides and offsets in values. Defaults: contiguous x [M, K], W [N, K], y [M, N]. kSplit: K slices (splitK kernels). */
export type MatmulLayout = {
  xStride?: number;
  wStride?: number;
  yStride?: number;
  xOffset?: number;
  wOffset?: number;
  yOffset?: number;
  kSplit?: number;
};

/** Params buffer for matmulWgsl. */
export function matmulParams(
  { m, n, k }: MatmulShape,
  scale: number,
  layout: MatmulLayout = {},
  conv?: { width: number; cin: number },
): Uint32Array<ArrayBuffer> {
  const { xStride = k, wStride = k, yStride = n, xOffset = 0, wOffset = 0, yOffset = 0, kSplit = 1 } = layout;
  for (const [name, v] of Object.entries({ K: k, xStride, wStride, xOffset, wOffset })) {
    if (v % 4) throw new Error(`${name} must be a multiple of 4, got ${v}`);
  }
  const params = new Uint32Array(13);
  params.set([m, n, k, 0, xStride, wStride, yStride, xOffset, wOffset, yOffset, conv?.width ?? 0, conv?.cin ?? 0, kSplit]);
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

/** One-shot run with bf16 W [N, K] given as raw little-endian bytes (2 per value). */
export async function matmulBf16(device: GPUDevice, x: Float32Array, w: Uint8Array, scale: number, shape: MatmulShape) {
  const { m, n, k } = shape;
  if (x.length !== m * k || w.length !== n * k * 2) throw new Error("x or w does not match the shape");
  const [y] = await runCompute(device, {
    code: matmulBf16Wgsl,
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

/** 3×3 conv, padding 1: image [height · width, cin] (channel-last) → [height · width, cout]. W: [cout, 9 · cin]. */
export function conv3x3Params(height: number, width: number, cin: number, cout: number) {
  if (cin % 4) throw new Error(`cin must be a multiple of 4, got ${cin}`);
  const shape = { m: height * width, n: cout, k: 9 * cin };
  return { shape, params: matmulParams(shape, 1, {}, { width, cin }) };
}

/** One-shot 3×3 conv run (tests, debugging). */
export async function conv3x3(device: GPUDevice, x: Float32Array, w: Float32Array, height: number, width: number, cin: number, cout: number) {
  const { shape, params } = conv3x3Params(height, width, cin, cout);
  const [y] = await runCompute(device, { code: conv3x3Wgsl, inputs: [params, x, w], outputs: [shape.m * cout * 4], workgroups: matmulWorkgroups(shape) });
  return new Float32Array(y);
}
