import { describe, expect, it } from "vitest";
import { e4m3, e5m2, toFloat32 } from "../src/weights/fp8";

describe("e4m3 (float8_e4m3fn)", () => {
  it.each([
    [0x00, 0],
    [0x38, 1],
    [0xc0, -2],
    [0x7e, 448], // largest
    [0x01, 2 ** -9], // smallest subnormal
    [0x08, 2 ** -6], // smallest normal
    [0x3c, 1.5],
  ])("0x%s → %s", (byte, value) => expect(e4m3(byte)).toBe(value));

  it("has negative zero", () => expect(Object.is(e4m3(0x80), -0)).toBe(true));
  it("has NaN and no infinity", () => {
    expect(e4m3(0x7f)).toBeNaN();
    expect(e4m3(0xff)).toBeNaN();
  });
});

describe("e5m2 (float8_e5m2)", () => {
  it.each([
    [0x00, 0],
    [0x3c, 1],
    [0x7b, 57344], // largest
    [0x01, 2 ** -16], // smallest subnormal
    [0xc0, -2],
  ])("0x%s → %s", (byte, value) => expect(e5m2(byte)).toBe(value));

  it("has infinity and NaN", () => {
    expect(e5m2(0x7c)).toBe(Infinity);
    expect(e5m2(0xfc)).toBe(-Infinity);
    expect(e5m2(0x7e)).toBeNaN();
  });
});

describe("toFloat32", () => {
  const le = (...words: number[]) => new Uint8Array(new Uint16Array(words).buffer);

  it("decodes F32", () => expect(toFloat32("F32", new Uint8Array(new Float32Array([1.5, -2]).buffer))).toEqual(new Float32Array([1.5, -2])));
  it("decodes BF16", () => expect(toFloat32("BF16", le(0x3f80, 0xc000))).toEqual(new Float32Array([1, -2])));
  it("decodes F16, incl. subnormals and infinity", () =>
    expect(toFloat32("F16", le(0x3c00, 0xc000, 0x0001, 0x7c00))).toEqual(new Float32Array([1, -2, 2 ** -24, Infinity])));
  it("decodes F8_E4M3 and F8_E5M2", () => {
    expect(toFloat32("F8_E4M3", new Uint8Array([0x38, 0x7e]))).toEqual(new Float32Array([1, 448]));
    expect(toFloat32("F8_E5M2", new Uint8Array([0x3c]))).toEqual(new Float32Array([1]));
  });
  it("refuses other dtypes", () => expect(() => toFloat32("I64", new Uint8Array(8))).toThrow(/I64/));
});
