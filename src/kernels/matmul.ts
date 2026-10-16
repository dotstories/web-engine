import { runCompute } from "../gpu/compute";
import { fp8Wgsl } from "./fp8";

/**
 * y[M, N] = x[M, K] · Wᵀ × scale, with W stored like PyTorch's nn.Linear: [N, K], fp8 e4m3fn,
 * packed 4 per u32 (straight from the file). Accumulates in f32.
 *
 * Tiles: 64×64 outputs per workgroup (16×16 threads × 4×4 each), K in steps of 32.
 * Needs K % 4 == 0 so every weight row starts on a u32 (true for every Krea 2 layer).
 */
export const TILE = 64;

export const matmulFp8Wgsl = /* wgsl */ `
${fp8Wgsl}
struct Params { m: u32, n: u32, k: u32, scale: f32 }

@group(0) @binding(0) var<storage, read> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

const TM = 64u;
const TN = 64u;
const TK = 32u;
var<workgroup> xs: array<array<f32, TK>, TM>;
var<workgroup> ws: array<array<f32, TK>, TN>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u,
        @builtin(local_invocation_index) li: u32) {
  let row0 = wg.y * TM;
  let col0 = wg.x * TN;
  var acc: array<array<f32, 4>, 4>;

  for (var k0 = 0u; k0 < p.k; k0 += TK) {
    // x tile: 64 × 32 floats, 8 per thread
    for (var i = 0u; i < 8u; i++) {
      let e = li + i * 256u;
      let r = row0 + e / TK;
      let c = k0 + e % TK;
      xs[e / TK][e % TK] = select(0.0, x[min(r * p.k + c, arrayLength(&x) - 1u)], r < p.m && c < p.k);
    }
    // W tile: 64 rows × 32 fp8 = 8 u32 per row, 2 words per thread, decoded once
    for (var i = 0u; i < 2u; i++) {
      let e = li + i * 256u;
      let r = e / 8u;
      let q = e % 8u;
      let n = col0 + r;
      let k = k0 + q * 4u;
      var word = 0u;
      if (n < p.n && k < p.k) { word = w[(n * p.k + k) / 4u]; }
      for (var b = 0u; b < 4u; b++) {
        ws[r][q * 4u + b] = select(0.0, e4m3_to_f32(fp8_byte(word, b)), n < p.n && k + b < p.k);
      }
    }
    workgroupBarrier();

    for (var kk = 0u; kk < TK; kk++) {
      var a: array<f32, 4>;
      var bv: array<f32, 4>;
      for (var i = 0u; i < 4u; i++) {
        a[i] = xs[lid.y * 4u + i][kk];
        bv[i] = ws[lid.x * 4u + i][kk];
      }
      for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) { acc[i][j] += a[i] * bv[j]; }
      }
    }
    workgroupBarrier();
  }

  for (var i = 0u; i < 4u; i++) {
    for (var j = 0u; j < 4u; j++) {
      let r = row0 + lid.y * 4u + i;
      let c = col0 + lid.x * 4u + j;
      if (r < p.m && c < p.n) { y[r * p.n + c] = acc[i][j] * p.scale; }
    }
  }
}
`;

export type MatmulShape = { m: number; n: number; k: number };

/** Params buffer for matmulFp8Wgsl. */
export function matmulParams({ m, n, k }: MatmulShape, scale: number): Uint32Array {
  if (k % 4) throw new Error(`K must be a multiple of 4, got ${k}`);
  const params = new Uint32Array(4);
  params.set([m, n, k]);
  new Float32Array(params.buffer)[3] = scale;
  return params;
}

/** One-shot run (tests, debugging). The engine keeps weights on the GPU instead. */
export async function matmulFp8(device: GPUDevice, x: Float32Array, w: Uint8Array, scale: number, shape: MatmulShape) {
  const { m, n, k } = shape;
  if (x.length !== m * k || w.length !== n * k) throw new Error("x or w does not match the shape");
  const packed = new Uint8Array(Math.ceil(w.length / 4) * 4);
  packed.set(w);
  const [y] = await runCompute(device, {
    code: matmulFp8Wgsl,
    inputs: [matmulParams(shape, scale), x, new Uint32Array(packed.buffer)],
    outputs: [m * n * 4],
    workgroups: [Math.ceil(n / TILE), Math.ceil(m / TILE)],
  });
  return new Float32Array(y);
}
