/** Small helpers to run one compute shader. Used by kernel tests and as the base of the engine. */

let device: Promise<GPUDevice> | undefined;

/** One shared device: the fast GPU, 16-bit shaders when available, the biggest buffers allowed. */
export function getDevice(): Promise<GPUDevice> {
  device ??= (async () => {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU adapter");
    return adapter.requestDevice({
      requiredFeatures: adapter.features.has("shader-f16") ? ["shader-f16"] : [],
      requiredLimits: {
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      },
    });
  })();
  return device;
}

export type Job = {
  code: string;
  entry?: string;
  inputs: ArrayBufferView[]; // bindings 0..n-1, read-only storage
  outputs: number[]; // byte sizes; bindings after the inputs, read_write storage
  workgroups: [number, number?, number?];
};

const pad4 = (n: number) => Math.max(4, Math.ceil(n / 4) * 4);

async function compile(device: GPUDevice, code: string) {
  const module = device.createShaderModule({ code });
  const errors = (await module.getCompilationInfo()).messages.filter((m) => m.type === "error");
  if (errors.length) throw new Error(`Shader error: ${errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join("; ")}`);
  return module;
}

/** Uploads the inputs, runs the shader once, returns each output's bytes. */
export async function runCompute(device: GPUDevice, job: Job): Promise<ArrayBuffer[]> {
  device.pushErrorScope("validation");
  const module = await compile(device, job.code).catch(async (e) => {
    await device.popErrorScope();
    throw e;
  });
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: job.entry ?? "main" } });

  const inputs = job.inputs.map((data) => {
    const buffer = device.createBuffer({ size: pad4(data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
    return buffer;
  });
  const outputs = job.outputs.map((size) =>
    device.createBuffer({ size: pad4(size), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }),
  );
  const reads = job.outputs.map((size) => device.createBuffer({ size: pad4(size), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }));

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [...inputs, ...outputs].map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(job.workgroups[0], job.workgroups[1] ?? 1, job.workgroups[2] ?? 1);
  pass.end();
  outputs.forEach((buffer, i) => encoder.copyBufferToBuffer(buffer, 0, reads[i], 0, buffer.size));
  device.queue.submit([encoder.finish()]);

  const error = await device.popErrorScope();
  try {
    if (error) throw new Error(`WebGPU error: ${error.message}`);
    return await Promise.all(
      reads.map(async (read, i) => {
        await read.mapAsync(GPUMapMode.READ);
        return read.getMappedRange().slice(0, job.outputs[i]);
      }),
    );
  } finally {
    [...inputs, ...outputs, ...reads].forEach((b) => b.destroy());
  }
}
