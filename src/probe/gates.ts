export const MB = 1024 ** 2;
export const GB = 1024 ** 3;

export type Report = {
  adapter: { vendor: string; architecture: string; device: string; description: string } | null;
  features: string[];
  maxStorageBufferBindingSize: number;
  maxBufferSize: number;
  gpuMemoryBytes: number | null; // allocated before out-of-memory (capped)
  quotaBytes: number | null;
  opfsReadBytesPerSec: number | null;
  uploadBytesPerSec: number | null;
  persisted: boolean | null; // persistent storage granted (lifts the per-site limit in some browsers)
  errors: Record<string, string>; // why a measurement failed, by gate id
};

export type Gate = { id: string; label: string; need: string; value: string; pass: boolean };

export function formatBytes(bytes: number): string {
  return bytes >= GB ? `${+(bytes / GB).toFixed(1)} GB` : `${Math.round(bytes / MB)} MB`;
}

const missing = (error?: string) => (error ? `failed: ${error}` : "not measured");

/** What Krea 2 in the browser needs (see the plan's W0 gate). */
export function evaluate(r: Report): Gate[] {
  const atLeast = (id: string, label: string, value: number | null, need: number, unit = "") => ({
    id,
    label,
    need: `≥ ${formatBytes(need)}${unit}`,
    value: value === null ? missing(r.errors[id]) : `${formatBytes(value)}${unit}`,
    pass: value !== null && value >= need,
  });
  const f16 = r.features.includes("shader-f16");
  return [
    { id: "webgpu", label: "WebGPU GPU", need: "found", value: r.adapter?.vendor || (r.adapter ? "unknown" : "none"), pass: !!r.adapter },
    { id: "f16", label: "16-bit shaders", need: "shader-f16", value: f16 ? "yes" : "no", pass: f16 },
    atLeast("binding", "Largest layer buffer", r.maxStorageBufferBindingSize, 256 * MB),
    atLeast("memory", "GPU memory", r.gpuMemoryBytes, 8 * GB),
    atLeast("quota", "Browser storage", r.quotaBytes, 20 * GB),
    atLeast("read", "Storage read speed", r.opfsReadBytesPerSec, GB, "/s"),
  ];
}
