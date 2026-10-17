import { TILE, matmulFp8Wgsl, matmulParams, type MatmulShape } from "../kernels/matmul";

/**
 * Average milliseconds per fp8 matmul of this shape on this GPU.
 * Buffers start zeroed and stay on the GPU: values don't change the speed, so nothing is uploaded.
 */
export async function timeMatmul(
  device: GPUDevice,
  shape: MatmulShape,
  iterations = 5,
  kernel: { code: string; workgroups: [number, number] } = { code: matmulFp8Wgsl, workgroups: [Math.ceil(shape.n / TILE), Math.ceil(shape.m / TILE)] },
): Promise<number> {
  const { m, n, k } = shape;
  const module = device.createShaderModule({ code: kernel.code });
  const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
  const storage = (size: number) => device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const params = matmulParams(shape, 1);
  const buffers = [storage(params.byteLength), storage(m * k * 4), storage(n * k), storage(m * n * 4)];
  device.queue.writeBuffer(buffers[0], 0, params);
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });

  const run = async (count: number) => {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    for (let i = 0; i < count; i++) pass.dispatchWorkgroups(...kernel.workgroups);
    pass.end();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
  };

  try {
    await run(1); // warm up
    const start = performance.now();
    await run(iterations);
    return (performance.now() - start) / iterations;
  } finally {
    buffers.forEach((b) => b.destroy());
  }
}
