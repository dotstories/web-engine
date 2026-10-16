import { runCompute } from "../gpu/compute";

/**
 * The small per-value steps of a Krea 2 block, in f32:
 * - swiglu:      out = silu(gate) · up        (MLP, before "down")
 * - sigmoidGate: out = a · sigmoid(gate)      (attention output gate, before "wo")
 * - gatedAdd:    out = x + g[i % d] · h       (residual add with the block's pregate/postgate)
 * - addBias:     out = x + b[i % d]           (bias of the bf16 linear layers around the blocks)
 * - geluTanh:    out = GELU(x), tanh form     (time and text MLPs)
 * - add:         out = x + h                  (plain residual in the text fusion blocks)
 * 256 values per workgroup; big tensors use the y dimension of the dispatch.
 */
const THREADS = 256;
const GROUPS_PER_DIM = 65535;

const kernel = (bindings: string, body: string) => /* wgsl */ `
struct Params { n: u32, d: u32 }
@group(0) @binding(0) var<storage, read> p: Params;
${bindings}

fn sigmoid(v: f32) -> f32 { return 1.0 / (1.0 + exp(-v)); }

// PyTorch GELU(approximate="tanh"); tanh written with exp so large |v| doesn't overflow
fn gelu_tanh(v: f32) -> f32 {
  let u = clamp(0.7978845608028654 * (v + 0.044715 * v * v * v), -15.0, 15.0);
  let e = exp(2.0 * u);
  return 0.5 * v * (1.0 + (e - 1.0) / (e + 1.0));
}

@compute @workgroup_size(${THREADS})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x + id.y * ${THREADS * GROUPS_PER_DIM}u;
  if (i >= p.n) { return; }
  ${body}
}`;

export const swigluWgsl = kernel(
  `@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read> up: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;`,
  "let g = gate[i];\n  out[i] = g * sigmoid(g) * up[i];",
);

export const sigmoidGateWgsl = kernel(
  `@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> gate: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;`,
  "out[i] = a[i] * sigmoid(gate[i]);",
);

export const gatedAddWgsl = kernel(
  `@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> h: array<f32>;
@group(0) @binding(3) var<storage, read> g: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;`,
  "out[i] = x[i] + g[i % p.d] * h[i];",
);

export const addBiasWgsl = kernel(
  `@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;`,
  "out[i] = x[i] + b[i % p.d];",
);

export const geluTanhWgsl = kernel(
  `@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;`,
  "out[i] = gelu_tanh(x[i]);",
);

export const addWgsl = kernel(
  `@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> h: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;`,
  "out[i] = x[i] + h[i];",
);

export function elementwiseWorkgroups(n: number): [number, number] {
  const groups = Math.ceil(n / THREADS);
  return [Math.min(groups, GROUPS_PER_DIM), Math.ceil(groups / GROUPS_PER_DIM)];
}

async function run(device: GPUDevice, code: string, [first, second, ...rest]: Float32Array[], d: number, sameLength = true) {
  const n = first.length;
  if (sameLength && second && second.length !== n) throw new Error("inputs must have the same length");
  const [out] = await runCompute(device, {
    code,
    inputs: [new Uint32Array([n, d]), first, ...(second ? [second] : []), ...rest],
    outputs: [n * 4],
    workgroups: elementwiseWorkgroups(n),
  });
  return new Float32Array(out);
}

/** One-shot runs (tests, debugging). The engine runs the same shaders on buffers that stay on the GPU. */
export const swiglu = (device: GPUDevice, gate: Float32Array, up: Float32Array) => run(device, swigluWgsl, [gate, up], 1);
export const sigmoidGate = (device: GPUDevice, a: Float32Array, gate: Float32Array) =>
  run(device, sigmoidGateWgsl, [a, gate], 1);
export const gatedAdd = (device: GPUDevice, x: Float32Array, h: Float32Array, g: Float32Array) =>
  run(device, gatedAddWgsl, [x, h, g], g.length);
export const addBias = (device: GPUDevice, x: Float32Array, b: Float32Array) => run(device, addBiasWgsl, [x, b], b.length, false);
export const geluTanh = (device: GPUDevice, x: Float32Array) => run(device, geluTanhWgsl, [x], 1);
export const add = (device: GPUDevice, x: Float32Array, h: Float32Array) => run(device, addWgsl, [x, h], 1);
