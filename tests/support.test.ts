import { describe, expect, it } from "vitest";
import { checkWebGPU } from "../src/gpu/support";

const adapter = { info: { vendor: "nvidia", architecture: "ada", device: "", description: "" } } as unknown as GPUAdapter;

describe("checkWebGPU", () => {
  it("says when the browser has no WebGPU", async () => {
    expect(await checkWebGPU(undefined)).toEqual({ ok: false, reason: "no-webgpu" });
  });

  it("says when there is no GPU adapter", async () => {
    const gpu = { requestAdapter: async () => null } as unknown as GPU;
    expect(await checkWebGPU(gpu)).toEqual({ ok: false, reason: "no-adapter" });
  });

  it("asks for the fast GPU and returns the adapter", async () => {
    let asked: GPURequestAdapterOptions | undefined;
    const gpu = {
      requestAdapter: async (o?: GPURequestAdapterOptions) => {
        asked = o;
        return adapter;
      },
    } as unknown as GPU;
    expect(await checkWebGPU(gpu)).toEqual({ ok: true, adapter });
    expect(asked).toEqual({ powerPreference: "high-performance" });
  });

  it("treats a failing request as no adapter", async () => {
    const gpu = { requestAdapter: async () => Promise.reject(new Error("blocked")) } as unknown as GPU;
    expect(await checkWebGPU(gpu)).toEqual({ ok: false, reason: "no-adapter" });
  });
});
