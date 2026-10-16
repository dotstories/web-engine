/**
 * fp8 (e4m3fn) weights in WGSL. WebGPU has no 8-bit floats, so weights stay packed 4 per u32
 * (little-endian, as in the file) and are decoded in the shader.
 */

/** WGSL helpers: byte i of a packed array, and e4m3fn → f32 (exact; subnormals without f32 denormals). */
export const fp8Wgsl = /* wgsl */ `
fn fp8_byte(packed: u32, i: u32) -> u32 {
  return (packed >> ((i & 3u) * 8u)) & 0xffu;
}

fn e4m3_to_f32(b: u32) -> f32 {
  let e = (b >> 3u) & 0xfu;
  let m = b & 7u;
  // normal: (1 + m/8) * 2^(e-7): rebias the exponent (7 -> 127) and move the mantissa to the top
  let normal = bitcast<f32>(((e + 120u) << 23u) | (m << 20u));
  let subnormal = f32(m) * 0.001953125; // m/8 * 2^-6
  let v = select(normal, subnormal, e == 0u);
  return select(v, -v, (b & 0x80u) != 0u);
}
`;

/** Test/utility kernel: out[i] = e4m3(bytes[i]) * scale. Bindings: packed bytes, scale, out. */
export const unpackFp8 = /* wgsl */ `
${fp8Wgsl}
@group(0) @binding(0) var<storage, read> packed: array<u32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= arrayLength(&out)) { return; }
  out[i] = e4m3_to_f32(fp8_byte(packed[i / 4u], i)) * scale[0];
}
`;
