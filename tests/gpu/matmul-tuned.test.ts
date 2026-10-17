import { expect, it } from "vitest";
import { getDevice, runCompute } from "../../src/gpu/compute";
import { matmulParams } from "../../src/kernels/matmul";
import { TUNE_CONFIGS, tunedSmemBytes, tunedWgsl, tunedWorkgroups, tuneName } from "../../src/kernels/matmul-tuned";
import { e4m3 } from "../../src/weights/fp8";

let seed = 3;
const rand = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32;
const [m, n, k] = [150, 260, 104]; // several tiles, ragged edges, K not a multiple of 32
const x = Float32Array.from({ length: m * k }, () => rand() * 2 - 1);
const w = Uint8Array.from({ length: n * k }, () => {
  const b = Math.floor(rand() * 256);
  return (b & 0x7f) === 0x7f ? b - 1 : b;
});
const ref = new Float64Array(m * n);
for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) for (let t = 0; t < k; t++) ref[i * n + j] += x[i * k + t] * e4m3(w[j * k + t]) * 0.5;

for (const config of TUNE_CONFIGS) {
  it(`tuned matmul ${tuneName(config)} matches the float64 reference`, async () => {
    const device = await getDevice();
    if (tunedSmemBytes(config) > device.limits.maxComputeWorkgroupStorageSize) return; // this GPU can't run it
    const [y] = await runCompute(device, {
      code: tunedWgsl(config),
      inputs: [matmulParams({ m, n, k }, 0.5), x, new Uint32Array(w.buffer)],
      outputs: [m * n * 4],
      workgroups: tunedWorkgroups(config, { m, n }),
    });
    const out = new Float32Array(y);
    let err = 0;
    let max = 0;
    for (let i = 0; i < ref.length; i++) {
      err = Math.max(err, Math.abs(out[i] - ref[i]));
      max = Math.max(max, Math.abs(ref[i]));
    }
    expect(err / max).toBeLessThan(1e-5);
  });
}
