/**
 * GPU peak: how many math operations per second a WebGPU shader can reach here, with no memory traffic.
 * Tells how far the matmul is from the ceiling, and whether f16 or int8 (dot4I8Packed) math would pay off.
 *
 * Every thread runs PEAK_CHAINS independent chains for `iters` iterations; each iteration is 64 operations
 * per thread: f32/f16 do 8 × vec4 fma (2 ops each), i8 does 8 × dot4I8Packed (4 multiply-adds).
 */
export type PeakKind = "f32" | "f16" | "i8";
export const PEAK_CHAINS = 8;
export const OPS_PER_ITER = 64;
const THREADS = 64;

const range = (n: number) => Array.from({ length: n }, (_, i) => i);

export function peakWgsl(kind: PeakKind): string {
  const chains = range(PEAK_CHAINS);
  const float = kind === "f16" ? "f16" : "f32";
  const body =
    kind === "i8"
      ? {
          init: chains.map((j) => `  var a${j} = i32(id.x % 16u) + ${j};`).join("\n"),
          step: chains.map((j) => `    a${j} += dot4I8Packed(bitcast<u32>(a${j}), y);`).join("\n"),
          sum: `bitcast<u32>(${chains.map((j) => `a${j}`).join(" + ")})`,
        }
      : {
          init: chains.map((j) => `  var a${j} = vec4<${float}>(${float}(id.x % 16u) + ${j}.0);`).join("\n"),
          step: chains.map((j) => `    a${j} = fma(a${j}, b, c);`).join("\n"),
          sum: `bitcast<u32>(dot(vec4<f32>(${chains.map((j) => `a${j}`).join(" + ")}), vec4<f32>(1.0)))`,
        };
  return /* wgsl */ `
${kind === "f16" ? "enable f16;" : ""}
struct Params { iters: u32, b: f32, c: f32, y: u32 }
@group(0) @binding(0) var<storage, read> p: Params;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(${THREADS})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let b = vec4<${float}>(${float}(p.b));
  let c = vec4<${float}>(${float}(p.c));
  let y = p.y;
${body.init}
  for (var i = 0u; i < p.iters; i++) {
${body.step}
  }
  out[id.x] = ${body.sum};
}
`;
}

/** iters, then b = 0.5 and c = 0.25 (floats stay exact), y = int8 bytes [1, -1, 2, 3]. */
export function peakParams(iters: number): Uint32Array<ArrayBuffer> {
  const params = new Uint32Array([iters, 0, 0, 0x0302ff01]);
  const f = new Float32Array(params.buffer);
  [f[1], f[2]] = [0.5, 0.25];
  return params;
}

export function peakSupported(device: GPUDevice, kind: PeakKind): boolean {
  if (kind === "f16") return device.features.has("shader-f16");
  if (kind === "i8") return navigator.gpu.wgslLanguageFeatures.has("packed_4x8_integer_dot_product");
  return true;
}

/** Average ms per dispatch and the operations per second (TFLOPS, or TOPS for i8). */
export async function timePeak(device: GPUDevice, kind: PeakKind, { workgroups = 16384, iters = 8192, runs = 5 } = {}) {
  const module = device.createShaderModule({ code: peakWgsl(kind) });
  const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
  const params = peakParams(iters);
  const paramsBuffer = device.createBuffer({ size: params.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const out = device.createBuffer({ size: workgroups * THREADS * 4, usage: GPUBufferUsage.STORAGE });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [paramsBuffer, out].map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  const run = async (count: number) => {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    for (let i = 0; i < count; i++) pass.dispatchWorkgroups(workgroups);
    pass.end();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
  };
  try {
    await run(1); // warm up
    const start = performance.now();
    await run(runs);
    const ms = (performance.now() - start) / runs;
    return { ms, tflops: (workgroups * THREADS * iters * OPS_PER_ITER) / (ms * 1e9) };
  } finally {
    paramsBuffer.destroy();
    out.destroy();
  }
}
