/**
 * Chains compute shaders on buffers that stay on the GPU: upload once, run many steps, read once.
 * Pipelines are compiled once per shader. Dispatches run in order, each one seeing the writes of
 * the ones before it.
 *
 * Memory: every buffer made here is freed by destroy(), or earlier with free() / release().
 * release(mark) frees the temporaries of one step (e.g. one transformer block) once it is queued.
 */

/** A whole buffer, or a part of it (offset and size in bytes; offsets must be multiples of 256). */
export type Binding = GPUBuffer | { buffer: GPUBuffer; offset: number; size: number };

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

  /** Part of a buffer, to bind e.g. one of the 6 modulation vectors. */
  static slice(buffer: GPUBuffer, offset: number, size: number): Binding {
    return { buffer, offset, size };
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

  private beginPass(): GPUComputePassEncoder {
    this.encoder ??= this.device.createCommandEncoder();
    this.pass ??= this.encoder.beginComputePass();
    return this.pass;
  }

  /** Queues one dispatch. Bindings in order: 0, 1, 2… (params usually first). */
  run(code: string, bindings: Binding[], workgroups: [number, number?, number?]): void {
    const pipeline = this.pipeline(code);
    const pass = this.beginPass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: bindings.map((b, binding) => ({ binding, resource: b instanceof GPUBuffer ? { buffer: b } : b })),
      }),
    );
    pass.dispatchWorkgroups(workgroups[0], workgroups[1] ?? 1, workgroups[2] ?? 1);
  }

  /** Queues a copy of `size` bytes, in order with the dispatches. */
  copy(src: GPUBuffer, srcOffset: number, dst: GPUBuffer, dstOffset: number, size: number): void {
    this.pass?.end();
    this.pass = null;
    this.encoder ??= this.device.createCommandEncoder();
    this.encoder.copyBufferToBuffer(src, srcOffset, dst, dstOffset, size);
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

  /** How many buffers exist now; pass it to release() later. */
  mark(): number {
    return this.owned.length;
  }

  /** Queues the work so far and frees every buffer made since `mark`, except `keep`. */
  release(mark: number, keep: GPUBuffer[] = []): void {
    this.submit(); // the GPU keeps freed buffers alive until the submitted work is done
    const later = this.owned.splice(mark);
    later.filter((b) => !keep.includes(b)).forEach((b) => b.destroy());
    this.owned.push(...later.filter((b) => keep.includes(b)));
  }

  /** Frees one buffer (after the work using it is queued). */
  free(buffer: GPUBuffer): void {
    this.submit();
    this.owned = this.owned.filter((b) => b !== buffer);
    buffer.destroy();
  }

  /** Frees every buffer made by this graph. */
  destroy(): void {
    this.submit();
    this.owned.forEach((b) => b.destroy());
    this.owned = [];
  }
}
