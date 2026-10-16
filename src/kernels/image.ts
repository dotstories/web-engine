import { runCompute } from "../gpu/compute";

/** Nearest ×2 upsample of a channel-last image [H · W, C] → [2H · 2W, C] (PyTorch "nearest-exact" at ×2). */
export const upsample2xWgsl = /* wgsl */ `
struct Params { height: u32, width: u32, channels: u32 }
@group(0) @binding(0) var<storage, read> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x + id.y * 256u * 65535u;
  let w2 = p.width * 2u;
  if (i >= p.height * 2u * w2 * p.channels) { return; }
  let c = i % p.channels;
  let pixel = i / p.channels;
  let x2 = pixel % w2;
  let y2 = pixel / w2;
  out[i] = x[((y2 / 2u) * p.width + x2 / 2u) * p.channels + c];
}
`;

export const upsample2xWorkgroups = (height: number, width: number, channels: number): [number, number] => {
  const groups = Math.ceil((4 * height * width * channels) / 256);
  return [Math.min(groups, 65535), Math.ceil(groups / 65535)];
};

export async function upsample2x(device: GPUDevice, x: Float32Array, height: number, width: number, channels: number) {
  const [out] = await runCompute(device, {
    code: upsample2xWgsl,
    inputs: [new Uint32Array([height, width, channels]), x],
    outputs: [x.byteLength * 4],
    workgroups: upsample2xWorkgroups(height, width, channels),
  });
  return new Float32Array(out);
}
