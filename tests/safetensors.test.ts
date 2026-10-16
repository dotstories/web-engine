import { describe, expect, it } from "vitest";
import { readHeader, readTensor, SafetensorsError } from "../src/weights/safetensors";
import { makeSafetensors, rawSafetensors } from "./helpers";

const bytes = (...b: number[]) => new Uint8Array(b);

describe("readHeader", () => {
  it("lists tensors with absolute file offsets", async () => {
    const file = makeSafetensors(
      { a: { dtype: "F8_E4M3", shape: [2, 2], data: bytes(1, 2, 3, 4) }, b: { dtype: "F32", shape: [1], data: bytes(0, 0, 128, 63) } },
      { format: "pt" },
    );
    const st = await readHeader(file);
    const start = file.size - 8; // data starts after size + header
    expect(st.metadata).toEqual({ format: "pt" });
    expect(st.tensors.get("a")).toEqual({ name: "a", dtype: "F8_E4M3", shape: [2, 2], offset: start, size: 4 });
    expect(st.tensors.get("b")).toMatchObject({ offset: start + 4, size: 4 });
  });

  it("reads one tensor's bytes without the rest of the file", async () => {
    const file = makeSafetensors({ a: { dtype: "U8", shape: [2], data: bytes(7, 8) }, b: { dtype: "U8", shape: [3], data: bytes(9, 10, 11) } });
    const st = await readHeader(file);
    expect(await readTensor(file, st.tensors.get("b")!)).toEqual(bytes(9, 10, 11));
  });

  it("accepts a file without tensors", async () => {
    expect((await readHeader(makeSafetensors({}))).tensors.size).toBe(0);
  });

  describe("rejects broken or hostile files", () => {
    const rejects = (file: Blob, message: RegExp) => expect(readHeader(file)).rejects.toThrow(message);

    it("too small", () => rejects(new Blob([bytes(1, 2)]), /too small/));
    it("header bigger than the file", () => rejects(rawSafetensors("{}", undefined, 1000n), /header size/));
    it("header over 100 MB", () => rejects(rawSafetensors("{}", undefined, 200_000_000n), /header size/));
    it("header is not JSON", () => rejects(rawSafetensors("nope"), /not valid JSON/));
    it("header is not an object", () => rejects(rawSafetensors("[1]"), /not an object/));
    it("unknown dtype", () => rejects(rawSafetensors('{"a":{"dtype":"X9","shape":[1],"data_offsets":[0,1]}}', bytes(0)), /unknown dtype/));
    it("bad shape", () => rejects(rawSafetensors('{"a":{"dtype":"U8","shape":[-1],"data_offsets":[0,1]}}', bytes(0)), /bad shape/));
    it("size does not match the shape", () =>
      rejects(rawSafetensors('{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}', bytes(0, 0, 0, 0)), /8 bytes/));
    it("data past the end of the file", () =>
      rejects(rawSafetensors('{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}', bytes(0)), /past the end/));
    it("overlapping tensors", () =>
      rejects(
        rawSafetensors(
          '{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[2],"data_offsets":[1,3]}}',
          bytes(0, 0, 0),
        ),
        /overlap/,
      ));
    it("errors are SafetensorsError", () => expect(readHeader(new Blob([]))).rejects.toBeInstanceOf(SafetensorsError));
  });
});

describe("readTensor with a limit", () => {
  it("reads only the first bytes of a tensor", async () => {
    const file = makeSafetensors({ a: { dtype: "U8", shape: [4], data: new Uint8Array([1, 2, 3, 4]) } });
    const st = await readHeader(file);
    expect(await readTensor(file, st.tensors.get("a")!, 2)).toEqual(new Uint8Array([1, 2]));
  });
});
