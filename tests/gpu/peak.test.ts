import { describe, expect, it } from "vitest";
import { getDevice, runCompute } from "../../src/gpu/compute";
import { PEAK_CHAINS, peakParams, peakSupported, peakWgsl, timePeak, type PeakKind } from "../../src/bench/peak";

/** CPU copy of one thread's chains (values chosen so f32/f16 stay exact). */
function cpuThread(kind: PeakKind, thread: number, iters: number): number {
  let sum = 0;
  for (let j = 0; j < PEAK_CHAINS; j++) {
    let a = (thread % 16) + j;
    for (let i = 0; i < iters; i++) {
      if (kind === "i8") {
        const bytes = [a & 0xff, (a >> 8) & 0xff, (a >> 16) & 0xff, (a >>> 24) & 0xff].map((b) => (b << 24) >> 24);
        const y = [1, -1, 2, 3]; // 0x0302ff01
        a = (a + bytes.reduce((s, b, k) => s + b * y[k], 0)) | 0;
      } else a = a * 0.5 + 0.25;
    }
    sum += kind === "i8" ? a : 4 * a; // float chains are vec4 with equal lanes
  }
  return kind === "i8" ? sum | 0 : sum;
}

describe("peak kernels", () => {
  it.each(["f32", "f16", "i8"] as const)("%s chains compute what the CPU does", async (kind) => {
    const device = await getDevice();
    if (!peakSupported(device, kind)) return; // f16 / packed dot product not on this GPU
    const iters = 3;
    const [out] = await runCompute(device, { code: peakWgsl(kind), inputs: [peakParams(iters)], outputs: [64 * 4], workgroups: [1] });
    const got = kind === "i8" ? new Int32Array(out) : new Float32Array(out);
    for (const t of [0, 5, 17, 63]) expect(got[t]).toBe(cpuThread(kind, t, iters));
  });

  it("times the f32 peak", async () => {
    const device = await getDevice();
    const { ms, tflops } = await timePeak(device, "f32", { workgroups: 4, iters: 16, warmupMs: 20 });
    expect(ms).toBeGreaterThan(0);
    expect(Number.isFinite(tflops)).toBe(true);
  });
});
