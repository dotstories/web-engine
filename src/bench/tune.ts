import { TUNE_CONFIGS, tunedSmemBytes, tunedWgsl, tunedWorkgroups, tuneName } from "../kernels/matmul-tuned";
import { timeMatmul } from "./matmul";

/** The three linear shapes that take ~95% of a block's matmul time, at 1928×1088 (8740 tokens). */
export const TUNE_SHAPES = [
  { name: "q / gate / out", m: 8740, n: 6144, k: 6144 },
  { name: "mlp gate / up", m: 8740, n: 16384, k: 6144 },
  { name: "mlp down", m: 8740, n: 6144, k: 16384 },
];

export type TuneResult = { variant: string; tflops: number[]; skipped?: string };

/** TFLOPS of every tunable matmul layout on each shape (variants the GPU can't run are skipped). */
export async function tuneMatmul(device: GPUDevice, shapes = TUNE_SHAPES, iterations = 3): Promise<TuneResult[]> {
  const results: TuneResult[] = [];
  for (const config of TUNE_CONFIGS) {
    const variant = tuneName(config);
    if (tunedSmemBytes(config) > device.limits.maxComputeWorkgroupStorageSize) {
      results.push({ variant, tflops: [], skipped: `needs ${tunedSmemBytes(config)} bytes of workgroup memory` });
      continue;
    }
    const tflops: number[] = [];
    for (const s of shapes) {
      const ms = await timeMatmul(device, s, iterations, { code: tunedWgsl(config), workgroups: tunedWorkgroups(config, s) });
      tflops.push((2 * s.m * s.n * s.k) / (ms * 1e9));
    }
    results.push({ variant, tflops });
  }
  return results;
}
