import { GB, MB, type Report } from "./gates";

export type Sizes = { memoryCap: number; memoryStep: number; upload: number; opfs: number; chunk: number };

export const FULL: Sizes = { memoryCap: 24 * GB, memoryStep: GB, upload: 256 * MB, opfs: GB, chunk: 16 * MB };
export const QUICK: Sizes = { memoryCap: 64 * MB, memoryStep: 16 * MB, upload: 16 * MB, opfs: 32 * MB, chunk: 8 * MB };

const now = () => performance.now() / 1000;

/** The biggest buffers the adapter allows, and 16-bit shaders when it has them. */
export function openDevice(adapter: GPUAdapter): Promise<GPUDevice> {
  return adapter.requestDevice({
    requiredFeatures: adapter.features.has("shader-f16") ? ["shader-f16"] : [],
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    },
  });
}

/** How much GPU memory we can allocate, in steps, up to `cap`. Everything is freed after. */
export async function measureGpuMemory(device: GPUDevice, step: number, cap: number): Promise<number> {
  const size = Math.min(step, device.limits.maxBufferSize) & ~3;
  const buffers: GPUBuffer[] = [];
  try {
    while ((buffers.length + 1) * size <= cap) {
      device.pushErrorScope("out-of-memory");
      const buffer = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(buffer, size - 4, new Uint32Array(1)); // touch it so it is really there
      if (await device.popErrorScope()) {
        buffer.destroy();
        break;
      }
      buffers.push(buffer);
    }
  } catch {
    // device lost: what we got so far is the answer
  }
  buffers.forEach((b) => b.destroy());
  return buffers.length * size;
}

/** CPU → GPU copy speed in bytes per second. */
export async function measureUpload(device: GPUDevice, bytes: number): Promise<number> {
  const buffer = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST });
  const data = new Uint8Array(bytes);
  device.queue.writeBuffer(buffer, 0, data, 0, 4); // warm up
  await device.queue.onSubmittedWorkDone();
  const start = now();
  device.queue.writeBuffer(buffer, 0, data);
  await device.queue.onSubmittedWorkDone();
  const seconds = now() - start;
  buffer.destroy();
  return bytes / seconds;
}

export async function measureQuota(): Promise<number | null> {
  return (await navigator.storage?.estimate?.())?.quota ?? null;
}

/** Browser storage (OPFS) read speed: write a test file, read it back, delete it. */
export async function measureOpfsRead(bytes: number, chunk: number): Promise<number> {
  const root = await navigator.storage.getDirectory();
  const name = "dotstories-probe.bin";
  try {
    const handle = await root.getFileHandle(name, { create: true });
    const writable = await handle.createWritable();
    const block = new Uint8Array(chunk).map((_, i) => i & 255);
    for (let written = 0; written < bytes; written += chunk) await writable.write(block);
    await writable.close();

    const reader = (await handle.getFile()).stream().getReader();
    const start = now();
    let read = 0;
    for (let r = await reader.read(); !r.done; r = await reader.read()) read += r.value.byteLength;
    return read / (now() - start);
  } finally {
    await root.removeEntry(name).catch(() => {});
  }
}

/** Runs a measurement; on failure keeps the reason in `errors[key]` and returns null. */
export async function attempt<T>(errors: Record<string, string>, key: string, f: () => Promise<T>): Promise<T | null> {
  try {
    return await f();
  } catch (e) {
    errors[key] = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
    return null;
  }
}

/** Ask to keep our files: browsers then allow more storage and don't evict it. */
export async function requestPersist(): Promise<boolean | null> {
  return navigator.storage?.persist ? navigator.storage.persist() : null;
}

export async function runProbe(adapter: GPUAdapter | null, sizes: Sizes, step: (s: string) => void): Promise<Report> {
  const report: Report = {
    adapter: adapter && {
      vendor: adapter.info.vendor,
      architecture: adapter.info.architecture,
      device: adapter.info.device,
      description: adapter.info.description,
    },
    features: adapter ? [...adapter.features].sort() : [],
    maxStorageBufferBindingSize: adapter?.limits.maxStorageBufferBindingSize ?? 0,
    maxBufferSize: adapter?.limits.maxBufferSize ?? 0,
    gpuMemoryBytes: null,
    quotaBytes: null,
    opfsReadBytesPerSec: null,
    uploadBytesPerSec: null,
    persisted: null,
    errors: {},
  };
  const { errors } = report;
  if (adapter) {
    const device = await openDevice(adapter);
    step("GPU memory");
    report.gpuMemoryBytes = await attempt(errors, "memory", () => measureGpuMemory(device, sizes.memoryStep, sizes.memoryCap));
    step("GPU upload speed");
    report.uploadBytesPerSec = await attempt(errors, "upload", () => measureUpload(device, sizes.upload));
    device.destroy();
  }
  step("Browser storage");
  report.persisted = await attempt(errors, "persist", requestPersist);
  report.quotaBytes = await attempt(errors, "quota", measureQuota);
  step("Storage read speed");
  report.opfsReadBytesPerSec = await attempt(errors, "read", () => measureOpfsRead(sizes.opfs, sizes.chunk));
  return report;
}
