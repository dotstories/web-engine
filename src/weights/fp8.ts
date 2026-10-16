/** CPU reference decoders. The GPU shaders do the same math; tests compare against these. */

/** float8_e4m3fn: bias 7, no infinity, 0x7f/0xff are NaN, max 448. */
export function e4m3(byte: number): number {
  const sign = byte & 0x80 ? -1 : 1;
  const exp = (byte >> 3) & 0xf;
  const man = byte & 0x7;
  if (exp === 0xf && man === 0x7) return NaN;
  if (exp === 0) return sign * (man / 8) * 2 ** -6;
  return sign * (1 + man / 8) * 2 ** (exp - 7);
}

/** float8_e5m2: bias 15, IEEE-style infinity and NaN. */
export function e5m2(byte: number): number {
  const sign = byte & 0x80 ? -1 : 1;
  const exp = (byte >> 2) & 0x1f;
  const man = byte & 0x3;
  if (exp === 0x1f) return man ? NaN : sign * Infinity;
  if (exp === 0) return sign * (man / 4) * 2 ** -14;
  return sign * (1 + man / 4) * 2 ** (exp - 15);
}

function f16(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exp = (bits >> 10) & 0x1f;
  const man = bits & 0x3ff;
  if (exp === 0x1f) return man ? NaN : sign * Infinity;
  if (exp === 0) return sign * (man / 1024) * 2 ** -14;
  return sign * (1 + man / 1024) * 2 ** (exp - 15);
}

const table = (decode: (b: number) => number) => Float32Array.from({ length: 256 }, (_, b) => decode(b));
const E4M3 = table(e4m3);
const E5M2 = table(e5m2);

/** Little-endian tensor bytes → Float32Array. */
export function toFloat32(dtype: string, bytes: Uint8Array): Float32Array {
  const words = () => new Uint16Array(bytes.slice().buffer);
  switch (dtype) {
    case "F32":
      return new Float32Array(bytes.slice().buffer);
    case "BF16":
      return new Float32Array(Uint32Array.from(words(), (w) => w << 16).buffer);
    case "F16":
      return Float32Array.from(words(), f16);
    case "F8_E4M3":
      return Float32Array.from(bytes, (b) => E4M3[b]);
    case "F8_E5M2":
      return Float32Array.from(bytes, (b) => E5M2[b]);
    default:
      throw new Error(`Can't decode ${dtype} as floats`);
  }
}
