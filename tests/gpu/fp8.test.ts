import { describe, expect, it } from "vitest";
import { getDevice, runCompute } from "../../src/gpu/compute";
import { unpackFp8 } from "../../src/kernels/fp8";
import { e4m3 } from "../../src/weights/fp8";

const NAN_CODES = new Set([0x7f, 0xff]); // never in model files; the GPU code doesn't special-case them

async function unpack(bytes: Uint8Array, scale: number) {
  const device = await getDevice();
  const padded = new Uint8Array(Math.ceil(bytes.length / 4) * 4);
  padded.set(bytes);
  const [out] = await runCompute(device, {
    code: unpackFp8,
    inputs: [new Uint32Array(padded.buffer), new Float32Array([scale])],
    outputs: [bytes.length * 4],
    workgroups: [Math.ceil(bytes.length / 64)],
  });
  return new Float32Array(out);
}

describe("unpackFp8 (e4m3fn → f32 × scale)", () => {
  it("decodes every byte exactly like the CPU reference", async () => {
    const all = Uint8Array.from({ length: 256 }, (_, i) => i);
    const gpu = await unpack(all, 1);
    for (const b of all) {
      if (!NAN_CODES.has(b)) expect([b, gpu[b]]).toEqual([b, e4m3(b)]);
    }
  });

  it("applies the per-tensor scale", async () => {
    const gpu = await unpack(new Uint8Array([0x38, 0xc0, 0x7e]), 0.0018833705689758062);
    expect(Array.from(gpu)).toEqual([1, -2, 448].map((v) => Math.fround(v * Math.fround(0.0018833705689758062))));
  });

  it("handles a length that is not a multiple of 4", async () => {
    expect(Array.from(await unpack(new Uint8Array([0x38, 0x38, 0x38, 0x38, 0x40]), 1))).toEqual([1, 1, 1, 1, 2]);
  });
});
