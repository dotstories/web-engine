/**
 * Chains compute shaders on buffers that stay on the GPU: upload once, run many steps, read once.
 * Pipelines are compiled once per shader. Dispatches run in order, each one seeing the writes of
 * the ones before it.
 */
export class Graph {
  private pipelines = new Map<string, GPUComputePipeline>();
  private encoder: GPUCommandEncoder | null = null;
  private pass: GPUComputePassEncoder | null = null;
  private owned: GPUBuffer[] = [];

  constructor(readonly device: GPUDevice) {}

  /** A storage buffer with this data, or zeroed with this many bytes. */
  buffer(data: ArrayBufferView | number): GPUBuffer {
    const size = typeof data === "number" ? data : data.byteLength;
    const buffer = this.device.createBuffer({
      size: Math.max(4, Math.ceil(size / 4) * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    // Typed arrays here never sit on a SharedArrayBuffer (no cross-origin isolation), so the cast is safe.
    if (typeof data !== "number") this.device.queue.writeBuffer(buffer, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
    this.owned.push(buffer);
    return buffer;
  }

  private pipeline(code: string): GPUComputePipeline {
    let pipeline = this.pipelines.get(code);
    if (!pipeline) {
      const module = this.device.createShaderModule({ code });
      pipeline = this.device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
      this.pipelines.set(code, pipeline);
    }
    return pipeline;
  }

  /** Queues one dispatch. Buffers bind in order: 0, 1, 2… (params usually first). */
  run(code: string, buffers: GPUBuffer[], workgroups: [number, number?, number?]): void {
    const pipeline = this.pipeline(code);
    this.encoder ??= this.device.createCommandEncoder();
    this.pass ??= this.encoder.beginComputePass();
    this.pass.setPipeline(pipeline);
    this.pass.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
      }),
    );
    this.pass.dispatchWorkgroups(workgroups[0], workgroups[1] ?? 1, workgroups[2] ?? 1);
  }

  /** Sends the queued work to the GPU. */
  submit(): void {
    this.pass?.end();
    if (this.encoder) this.device.queue.submit([this.encoder.finish()]);
    this.pass = null;
    this.encoder = null;
  }

  /** Runs everything queued, then copies `bytes` of `buffer` back. */
  async read(buffer: GPUBuffer, bytes = buffer.size): Promise<ArrayBuffer> {
    this.submit();
    const staging = this.device.createBuffer({ size: Math.ceil(bytes / 4) * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, staging.size);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const data = staging.getMappedRange().slice(0, bytes);
    staging.destroy();
    return data;
  }

  /** Frees every buffer made by this graph. */
  destroy(): void {
    this.submit();
    this.owned.forEach((b) => b.destroy());
    this.owned = [];
  }
}
