export type Support = { ok: true; adapter: GPUAdapter } | { ok: false; reason: "no-webgpu" | "no-adapter" };

/** Is WebGPU usable here? Asks for the fast GPU on laptops with two. */
export async function checkWebGPU(gpu: GPU | undefined = globalThis.navigator?.gpu): Promise<Support> {
  if (!gpu) return { ok: false, reason: "no-webgpu" };
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" }).catch(() => null);
  return adapter ? { ok: true, adapter } : { ok: false, reason: "no-adapter" };
}
