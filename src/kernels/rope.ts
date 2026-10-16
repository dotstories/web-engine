import { runCompute } from "../gpu/compute";

/**
 * Rotary position embedding as in Krea 2 (diffusers Krea2RotaryPosEmbed + apply_rotary_emb):
 * 3 axes (t, h, w) of 32, 48 and 48 dims, theta 1000, pairs interleaved in the head:
 *   out[2j]   = x[2j] · cos_j − x[2j+1] · sin_j
 *   out[2j+1] = x[2j+1] · cos_j + x[2j] · sin_j
 * cos/sin are computed on the CPU in float64 (like PyTorch) and uploaded as tables [tokens, pairs].
 */
export const AXES = [32, 48, 48];
export const THETA = 1000;

/** Text tokens at (0, 0, 0), then image tokens row by row at (0, row, column) on the latent grid. */
export function positions(textTokens: number, rows: number, columns: number): number[][] {
  const text = Array.from({ length: textTokens }, () => [0, 0, 0]);
  const image = Array.from({ length: rows * columns }, (_, i) => [0, Math.floor(i / columns), i % columns]);
  return [...text, ...image];
}

export function ropeTables(pos: number[][], axes = AXES, theta = THETA) {
  if (axes.some((d) => d % 2)) throw new Error("Every RoPE axis needs an even number of dims");
  const pairs = axes.reduce((sum, d) => sum + d / 2, 0);
  const cos = new Float32Array(pos.length * pairs);
  const sin = new Float32Array(pos.length * pairs);
  pos.forEach((p, t) => {
    let pair = 0;
    axes.forEach((dim, axis) => {
      for (let j = 0; j < dim / 2; j++, pair++) {
        const angle = p[axis] * theta ** (-(2 * j) / dim);
        cos[t * pairs + pair] = Math.cos(angle);
        sin[t * pairs + pair] = Math.sin(angle);
      }
    });
  });
  return { cos, sin, pairs };
}

/** x: [tokens, heads, 2 · pairs]. One thread per pair. */
export const ropeWgsl = /* wgsl */ `
struct Params { tokens: u32, heads: u32, pairs: u32 }
@group(0) @binding(0) var<storage, read> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> cosines: array<f32>;
@group(0) @binding(3) var<storage, read> sines: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<vec2<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let pair = id.x;
  let token = id.y + id.z * 65535u;
  if (pair >= p.pairs || token >= p.tokens) { return; }
  let c = cosines[token * p.pairs + pair];
  let s = sines[token * p.pairs + pair];
  for (var h = 0u; h < p.heads; h++) {
    let i = (token * p.heads + h) * p.pairs + pair;
    let v = x[i];
    out[i] = vec2<f32>(v.x * c - v.y * s, v.y * c + v.x * s);
  }
}
`;

export const ropeWorkgroups = (tokens: number, pairs: number): [number, number, number] => [
  Math.ceil(pairs / 64),
  Math.min(tokens, 65535),
  Math.ceil(tokens / 65535),
];

/** One-shot run (tests, debugging). */
export async function rope(device: GPUDevice, x: Float32Array, heads: number, tables: ReturnType<typeof ropeTables>) {
  const tokens = tables.cos.length / tables.pairs;
  if (x.length !== tokens * heads * tables.pairs * 2) throw new Error("x does not match tokens × heads × head dim");
  const [out] = await runCompute(device, {
    code: ropeWgsl,
    inputs: [new Uint32Array([tokens, heads, tables.pairs]), x, tables.cos, tables.sin],
    outputs: [x.byteLength],
    workgroups: ropeWorkgroups(tokens, tables.pairs),
  });
  return new Float32Array(out);
}
