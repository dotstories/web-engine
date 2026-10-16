import { runCompute } from "../gpu/compute";
import { fp8Wgsl } from "./fp8";

/**
 * y[M, N] = x[M, K] · Wᵀ × scale, with W stored like PyTorch's nn.Linear: [N, K], fp8 e4m3fn,
 * packed 4 per u32 (straight from the file). Accumulates in f32.
 *
 * Tiles: 128×128 outputs per workgroup (16×16 threads × 8×8 each), K in steps of 16.
 * x and W are read 4 values at a time; W is decoded once per tile into workgroup memory,
 * stored k-major so each thread reads 8 neighbours per step.
 * Needs K % 4 == 0 so every row starts on a u32 (true for every Krea 2 layer).
 */
export const TILE = 128;

export const matmulFp8Wgsl = /* wgsl */ `
${fp8Wgsl}
struct Params { m: u32, n: u32, k: u32, scale: f32 }

@group(0) @binding(0) var<storage, read> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

const TM = 128u;
const TN = 128u;
const TK = 16u;
var<workgroup> xs: array<f32, 2048>; // [TK][TM]
var<workgroup> ws: array<f32, 2048>; // [TK][TN]

fn x8(at: u32) -> array<f32, 8> {
  return array<f32, 8>(xs[at], xs[at + 1u], xs[at + 2u], xs[at + 3u], xs[at + 4u], xs[at + 5u], xs[at + 6u], xs[at + 7u]);
}

fn w8(at: u32) -> array<f32, 8> {
  return array<f32, 8>(ws[at], ws[at + 1u], ws[at + 2u], ws[at + 3u], ws[at + 4u], ws[at + 5u], ws[at + 6u], ws[at + 7u]);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u,
        @builtin(local_invocation_index) li: u32) {
  let row0 = wg.y * TM;
  let col0 = wg.x * TN;
  let k4 = p.k / 4u;
  var acc: array<vec4<f32>, 16>; // 8 rows × 2 vec4 (8 columns)

  for (var k0 = 0u; k0 < p.k; k0 += TK) {
    // Load: 128 rows × 4 groups of 4 k, for x and W. 2 per thread; 4 neighbouring threads read one row.
    for (var t = 0u; t < 2u; t++) {
      let e = li + t * 256u;
      let r = e / 4u;
      let q = e % 4u;
      let kq = k0 / 4u + q;
      let at = 4u * q * TM + r;

      var v = vec4<f32>(0.0);
      if (row0 + r < p.m && kq < k4) { v = x[(row0 + r) * k4 + kq]; }
      xs[at] = v.x;
      xs[at + TM] = v.y;
      xs[at + 2u * TM] = v.z;
      xs[at + 3u * TM] = v.w;

      var word = 0u; // decodes to 0.0
      if (col0 + r < p.n && kq < k4) { word = w[(col0 + r) * k4 + kq]; }
      ws[at] = e4m3_to_f32(fp8_byte(word, 0u));
      ws[at + TN] = e4m3_to_f32(fp8_byte(word, 1u));
      ws[at + 2u * TN] = e4m3_to_f32(fp8_byte(word, 2u));
      ws[at + 3u * TN] = e4m3_to_f32(fp8_byte(word, 3u));
    }
    workgroupBarrier();

    for (var kk = 0u; kk < TK; kk++) {
      let a = x8(kk * TM + lid.y * 8u);
      let b = w8(kk * TN + lid.x * 8u);
      let b0 = vec4<f32>(b[0], b[1], b[2], b[3]);
      let b1 = vec4<f32>(b[4], b[5], b[6], b[7]);
      for (var i = 0u; i < 8u; i++) {
        acc[i * 2u] += a[i] * b0;
        acc[i * 2u + 1u] += a[i] * b1;
      }
    }
    workgroupBarrier();
  }

  for (var i = 0u; i < 8u; i++) {
    let r = row0 + lid.y * 8u + i;
    if (r >= p.m) { break; }
    for (var j = 0u; j < 8u; j++) {
      let c = col0 + lid.x * 8u + j;
      if (c < p.n) { y[r * p.n + c] = acc[i * 2u + j / 4u][j % 4u] * p.scale; }
    }
  }
}
`;

export type MatmulShape = { m: number; n: number; k: number };

/** Params buffer for matmulFp8Wgsl. */
export function matmulParams({ m, n, k }: MatmulShape, scale: number): Uint32Array<ArrayBuffer> {
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
