import { describe, expect, it } from "vitest";
import { GB, MB, evaluate, formatBytes, type Report } from "../src/probe/gates";

const good: Report = {
  adapter: { vendor: "nvidia", architecture: "lovelace", device: "", description: "" },
  features: ["shader-f16", "timestamp-query"],
  maxStorageBufferBindingSize: 2 * GB,
  maxBufferSize: 4 * GB,
  gpuMemoryBytes: 14 * GB,
  quotaBytes: 500 * GB,
  opfsReadBytesPerSec: 3 * GB,
  uploadBytesPerSec: 10 * GB,
};

const gate = (r: Report, id: string) => evaluate(r).find((g) => g.id === id)!;

describe("evaluate", () => {
  it("passes every gate on a good machine", () => {
    expect(evaluate(good).every((g) => g.pass)).toBe(true);
  });

  it("checks the gates the plan needs, in order", () => {
    expect(evaluate(good).map((g) => g.id)).toEqual(["webgpu", "f16", "binding", "memory", "quota", "read"]);
  });

  it("fails without an adapter", () => {
    expect(gate({ ...good, adapter: null }, "webgpu").pass).toBe(false);
  });

  it("needs shader-f16", () => {
    expect(gate({ ...good, features: ["timestamp-query"] }, "f16").pass).toBe(false);
  });

  it("needs 256 MB storage bindings (the biggest layer)", () => {
    expect(gate({ ...good, maxStorageBufferBindingSize: 128 * MB }, "binding").pass).toBe(false);
    expect(gate({ ...good, maxStorageBufferBindingSize: 256 * MB }, "binding").pass).toBe(true);
  });

  it("needs 8 GB of GPU memory (int4 transformer + activations)", () => {
    expect(gate({ ...good, gpuMemoryBytes: 7 * GB }, "memory").pass).toBe(false);
    expect(gate({ ...good, gpuMemoryBytes: 8 * GB }, "memory").pass).toBe(true);
  });

  it("needs 20 GB of browser storage", () => {
    expect(gate({ ...good, quotaBytes: 19 * GB }, "quota").pass).toBe(false);
  });

  it("needs 1 GB/s reads from browser storage", () => {
    expect(gate({ ...good, opfsReadBytesPerSec: 0.5 * GB }, "read").pass).toBe(false);
  });

  it("fails a gate it could not measure", () => {
    const g = gate({ ...good, gpuMemoryBytes: null }, "memory");
    expect(g).toMatchObject({ pass: false, value: "not measured" });
  });
});

describe("formatBytes", () => {
  it("uses GB from 1 GB up", () => {
    expect(formatBytes(1.5 * GB)).toBe("1.5 GB");
  });
  it("uses MB below 1 GB", () => {
    expect(formatBytes(256 * MB)).toBe("256 MB");
  });
});
