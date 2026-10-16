import { describe, expect, it } from "vitest";
import { embed, LAYERS, TAPS } from "../src/model/text-encoder";
import type { TensorInfo } from "../src/weights/safetensors";

describe("text encoder layout", () => {
  it("taps 12 layers and runs only the 35 it needs", () => {
    expect(TAPS).toEqual([2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]);
    expect(LAYERS).toBe(35);
  });
});

describe("embed", () => {
  it("reads only the rows of the given token ids from a bf16 table", async () => {
    const [vocab, hidden] = [4, 2];
    const words = Uint16Array.from({ length: vocab * hidden }, (_, i) => new Uint32Array(new Float32Array([i]).buffer)[0] >>> 16);
    const file = new Blob([new Uint8Array(8), new Uint8Array(words.buffer)]); // table starts at byte 8
    const table: TensorInfo = { name: "e", dtype: "BF16", shape: [vocab, hidden], offset: 8, size: words.byteLength };
    expect(Array.from(await embed(file, table, [3, 0, 3]))).toEqual([6, 7, 0, 1, 6, 7]);
    await expect(embed(file, table, [4])).rejects.toThrow(/outside/);
  });
});
