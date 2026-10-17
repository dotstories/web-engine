import { describe, expect, it, vi } from "vitest";
import { abortable, generate } from "../src/pipeline/generate";

const { getDevice } = vi.hoisted(() => ({ getDevice: vi.fn() }));
vi.mock("../src/gpu/compute", () => ({ getDevice }));

type Files = Parameters<typeof generate>[0];
type Tokenizer = Parameters<typeof generate>[1];

const progress = { stage: "steps", message: "Step 1 of 8…", fraction: 0.3 } as const;

describe("abortable", () => {
  it("passes progress on until the signal is aborted, then throws an AbortError", () => {
    const seen: string[] = [];
    const controller = new AbortController();
    const report = abortable((p) => seen.push(p.message), controller.signal);
    report(progress);
    controller.abort();
    expect(() => report(progress)).toThrow(expect.objectContaining({ name: "AbortError" }));
    expect(seen).toEqual(["Step 1 of 8…"]);
  });

  it("never throws without a signal", () => {
    const report = abortable(() => {});
    expect(() => report(progress)).not.toThrow();
  });
});

describe("generate", () => {
  it("does not touch the GPU when cancelled before it starts", async () => {
    const controller = new AbortController();
    controller.abort();
    const run = generate({} as Files, {} as Tokenizer, { prompt: "x", width: 64, height: 64, seed: 1 }, () => {}, {
      signal: controller.signal,
    });
    await expect(run).rejects.toMatchObject({ name: "AbortError" });
    expect(getDevice).not.toHaveBeenCalled();
  });
});
