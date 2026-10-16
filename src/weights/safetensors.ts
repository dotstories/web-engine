/**
 * Reads .safetensors files (https://github.com/huggingface/safetensors) straight from a File or Blob.
 * Only the header is read up front; tensors are sliced out when needed, so a 13 GB file opens instantly.
 * The file is untrusted: every offset and size is checked before use.
 */

/** Bytes per element for each dtype we may meet in model files. */
export const DTYPE_BYTES: Record<string, number> = {
  F64: 8, F32: 4, F16: 2, BF16: 2, F8_E4M3: 1, F8_E5M2: 1,
  I64: 8, I32: 4, I16: 2, I8: 1, U64: 8, U32: 4, U16: 2, U8: 1, BOOL: 1,
};

const MAX_HEADER = 100_000_000; // same limit as the Python library

export type TensorInfo = { name: string; dtype: string; shape: number[]; offset: number; size: number };
export type Safetensors = { tensors: Map<string, TensorInfo>; metadata: Record<string, string> };

export class SafetensorsError extends Error {
  name = "SafetensorsError";
}

function fail(message: string): never {
  throw new SafetensorsError(message);
}

const isCount = (n: unknown): n is number => Number.isSafeInteger(n) && (n as number) >= 0;

export async function readHeader(file: Blob): Promise<Safetensors> {
  if (file.size < 8) fail("File is too small to be a .safetensors file");
  const headerSize = Number(new DataView(await file.slice(0, 8).arrayBuffer()).getBigUint64(0, true));
  if (headerSize > MAX_HEADER || 8 + headerSize > file.size) fail(`Bad header size: ${headerSize}`);

  let header: unknown;
  try {
    header = JSON.parse(await file.slice(8, 8 + headerSize).text());
  } catch {
    fail("Header is not valid JSON");
  }
  if (!header || typeof header !== "object" || Array.isArray(header)) fail("Header is not an object");

  const start = 8 + headerSize;
  const dataSize = file.size - start;
  const { __metadata__: metadata = {}, ...entries } = header as Record<string, unknown>;
  const tensors = new Map<string, TensorInfo>();

  for (const [name, entry] of Object.entries(entries)) {
    const { dtype, shape, data_offsets: offsets } = (entry ?? {}) as Record<string, unknown>;
    if (typeof dtype !== "string" || !(dtype in DTYPE_BYTES)) fail(`${name}: unknown dtype ${String(dtype)}`);
    if (!Array.isArray(shape) || !shape.every(isCount)) fail(`${name}: bad shape`);
    if (!Array.isArray(offsets) || offsets.length !== 2 || !offsets.every(isCount)) fail(`${name}: bad data_offsets`);
    const [begin, end] = offsets as number[];
    const expected = (shape as number[]).reduce((a, b) => a * b, 1) * DTYPE_BYTES[dtype as string];
    if (end - begin !== expected) fail(`${name}: has ${end - begin} bytes, its shape needs ${expected} bytes`);
    if (end > dataSize) fail(`${name}: data goes past the end of the file`);
    tensors.set(name, { name, dtype: dtype as string, shape: shape as number[], offset: start + begin, size: end - begin });
  }

  const sorted = [...tensors.values()].sort((a, b) => a.offset - b.offset);
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i].offset < sorted[i - 1].offset + sorted[i - 1].size) fail(`${sorted[i - 1].name} and ${sorted[i].name} overlap`);
  }

  return { tensors, metadata: metadata as Record<string, string> };
}

/** The raw bytes of one tensor (or only its first `limit` bytes). */
export async function readTensor(file: Blob, tensor: TensorInfo, limit = tensor.size): Promise<Uint8Array> {
  const end = tensor.offset + Math.min(limit, tensor.size);
  return new Uint8Array(await file.slice(tensor.offset, end).arrayBuffer());
}
