import { describe, expect, it } from "vitest";
import { getDevice, runCompute } from "../../src/gpu/compute";

const ADD = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x < arrayLength(&out)) { out[id.x] = a[id.x] + b[id.x]; }
}`;

describe("runCompute", () => {
  it("runs a shader on the GPU and reads the result back", async () => {
    const device = await getDevice();
    const n = 1000;
    const a = Float32Array.from({ length: n }, (_, i) => i);
    const b = Float32Array.from({ length: n }, () => 0.5);
    const [out] = await runCompute(device, { code: ADD, inputs: [a, b], outputs: [n * 4], workgroups: [Math.ceil(n / 64)] });
    expect(new Float32Array(out)).toEqual(Float32Array.from({ length: n }, (_, i) => i + 0.5));
  });

  it("reuses one device", async () => {
    expect(await getDevice()).toBe(await getDevice());
  });

  it("explains a shader that does not compile", async () => {
    const device = await getDevice();
    await expect(runCompute(device, { code: "fn main( {", inputs: [], outputs: [4], workgroups: [1] })).rejects.toThrow(/shader/i);
  });
});
