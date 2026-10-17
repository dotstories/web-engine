import { fp8Wgsl } from "./fp8";

/**
 * A tunable copy of the fp8 matmul (y[M, N] = x[M, K] · Wᵀ × scale, W fp8 [N, K]) for finding the fastest
 * layout on a GPU (/pages/bench/). Same params as matmulParams (contiguous rows; y stride and offset kept).
 *
 *   tm:   rows per workgroup: 128 (16×16 threads) or 64 (16×8 threads); always 128 columns, 8×8 per thread
 *   tk:   K per step between barriers: 16 or 32
 *   vec4: workgroup memory as vec4 (2 loads of 4 instead of 8 single loads per side and step)
 *
 * The production kernel (matmul.ts) is tm 128, tk 16, scalar.
 */
export type TuneConfig = { tm: 64 | 128; tk: 16 | 32; vec4: boolean };

export const TUNE_CONFIGS: TuneConfig[] = [128, 64].flatMap((tm) =>
  [16, 32].flatMap((tk) => [false, true].map((vec4) => ({ tm, tk, vec4 }) as TuneConfig)),
);

export const tuneName = ({ tm, tk, vec4 }: TuneConfig) => `${tm}×128 tile · K step ${tk} · ${vec4 ? "vec4" : "scalar"}`;

/** Workgroup memory bytes (checked against the device limit). */
export const tunedSmemBytes = ({ tm, tk }: TuneConfig) => (tm + 128) * tk * 4;

export const tunedWorkgroups = ({ tm }: TuneConfig, { m, n }: { m: number; n: number }): [number, number] => [Math.ceil(n / 128), Math.ceil(m / tm)];

const range = (n: number) => Array.from({ length: n }, (_, i) => i);
const lines = (n: number, line: (i: number) => string) => range(n).map(line).join("\n");

export function tunedWgsl({ tm, tk, vec4 }: TuneConfig): string {
  const threads = 16 * (tm / 8);
  const xLoads = (tm * tk) / 4 / threads; // vec4 of x per thread and step
  const wLoads = (128 * tk) / 4 / threads; // u32 (4 weights) per thread and step
  const q4 = tk / 4; // vec4 per row and step
  const sm = (name: string, size: number) =>
    vec4 ? `var<workgroup> ${name}: array<vec4<f32>, ${size / 4}>;` : `var<workgroup> ${name}: array<f32, ${size}>;`;
  const put = (name: string, at: string, v: string) => (vec4 ? `${name}[(${at}) / 4u][(${at}) % 4u] = ${v};` : `${name}[${at}] = ${v};`);
  const xLoad = (i: number) => /* wgsl */ `
    {
      let e = li + ${i * threads}u;
      let r = e / ${q4}u;
      let q = e % ${q4}u;
      let kq = k0 / 4u + q;
      var v = vec4<f32>(0.0);
      if (row0 + r < p.m && kq < k4) { v = x[(p.x_offset + (row0 + r) * p.x_stride) / 4u + kq]; }
      let at = 4u * q * TM + r;
      ${put("xs", "at", "v.x")}
      ${put("xs", "at + TM", "v.y")}
      ${put("xs", "at + 2u * TM", "v.z")}
      ${put("xs", "at + 3u * TM", "v.w")}
    }`;
  const wLoad = (i: number) => /* wgsl */ `
    {
      let e = li + ${i * threads}u;
      let r = e / ${q4}u;
      let q = e % ${q4}u;
      let kq = k0 / 4u + q;
      var word = 0u;
      if (col0 + r < p.n && kq < k4) { word = w[(p.w_offset + (col0 + r) * p.w_stride) / 4u + kq]; }
      let at = 4u * q * TN + r;
      ${put("ws", "at", "e4m3_to_f32(fp8_byte(word, 0u))")}
      ${put("ws", "at + TN", "e4m3_to_f32(fp8_byte(word, 1u))")}
      ${put("ws", "at + 2u * TN", "e4m3_to_f32(fp8_byte(word, 2u))")}
      ${put("ws", "at + 3u * TN", "e4m3_to_f32(fp8_byte(word, 3u))")}
    }`;
  const reads = vec4
    ? `      let av0 = xs[(kk * TM + lid.y * 8u) / 4u];
      let av1 = xs[(kk * TM + lid.y * 8u) / 4u + 1u];
${lines(8, (i) => `      let a${i} = av${i >> 2}.${"xyzw"[i & 3]};`)}
      let b0 = ws[(kk * TN + lid.x * 8u) / 4u];
      let b1 = ws[(kk * TN + lid.x * 8u) / 4u + 1u];`
    : `      let xa = kk * TM + lid.y * 8u;
      let wa = kk * TN + lid.x * 8u;
${lines(8, (i) => `      let a${i} = xs[xa + ${i}u];`)}
      let b0 = vec4<f32>(ws[wa], ws[wa + 1u], ws[wa + 2u], ws[wa + 3u]);
      let b1 = vec4<f32>(ws[wa + 4u], ws[wa + 5u], ws[wa + 6u], ws[wa + 7u]);`;

  return /* wgsl */ `
${fp8Wgsl}
struct Params {
  m: u32, n: u32, k: u32, scale: f32,
  x_stride: u32, w_stride: u32, y_stride: u32,
  x_offset: u32, w_offset: u32, y_offset: u32,
  width: u32, cin: u32, k_split: u32,
}

@group(0) @binding(0) var<storage, read> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

const TM = ${tm}u;
const TN = 128u;
const TK = ${tk}u;
${sm("xs", tm * tk)} // [TK][TM]
${sm("ws", 128 * tk)} // [TK][TN]

fn store4(r: u32, c: u32, v: vec4<f32>) {
  let o = p.y_offset + r * p.y_stride + c;
${lines(4, (i) => `  if (c + ${i}u < p.n) { y[o + ${i}u] = v.${"xyzw"[i]} * p.scale; }`)}
}

@compute @workgroup_size(16, ${tm / 8})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u,
        @builtin(local_invocation_index) li: u32) {
  let row0 = wg.y * TM;
  let col0 = wg.x * TN;
  let k4 = p.k / 4u;
${lines(16, (i) => `  var acc${i} = vec4<f32>(0.0);`)}

  for (var k0 = 0u; k0 < p.k; k0 += TK) {
${range(xLoads).map(xLoad).join("\n")}
${range(wLoads).map(wLoad).join("\n")}
    workgroupBarrier();

    for (var kk = 0u; kk < TK; kk++) {
${reads}
${lines(8, (i) => `      acc${2 * i} += a${i} * b0;\n      acc${2 * i + 1} += a${i} * b1;`)}
    }
    workgroupBarrier();
  }

  let c0 = col0 + lid.x * 8u;
${lines(8, (i) => `  if (row0 + lid.y * 8u + ${i}u < p.m) { store4(row0 + lid.y * 8u + ${i}u, c0, acc${2 * i}); store4(row0 + lid.y * 8u + ${i}u, c0 + 4u, acc${2 * i + 1}); }`)}
}
`;
}
