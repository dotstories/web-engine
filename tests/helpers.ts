/** Builds a .safetensors file in memory, like the Python library writes it. */
export function makeSafetensors(
  tensors: Record<string, { dtype: string; shape: number[]; data: Uint8Array<ArrayBuffer> }>,
  metadata?: Record<string, string>,
): Blob {
  const header: Record<string, unknown> = metadata ? { __metadata__: metadata } : {};
  let offset = 0;
  for (const [name, t] of Object.entries(tensors)) {
    header[name] = { dtype: t.dtype, shape: t.shape, data_offsets: [offset, offset + t.data.byteLength] };
    offset += t.data.byteLength;
  }
  const json = new TextEncoder().encode(JSON.stringify(header));
  const size = new Uint8Array(8);
  new DataView(size.buffer).setBigUint64(0, BigInt(json.byteLength), true);
  return new Blob([size, json, ...Object.values(tensors).map((t) => t.data)]);
}

/** A raw file: 8-byte size + the given header text + data. */
export function rawSafetensors(headerText: string, data: Uint8Array<ArrayBuffer> = new Uint8Array(), sizeOverride?: bigint): Blob {
  const json = new TextEncoder().encode(headerText);
  const size = new Uint8Array(8);
  new DataView(size.buffer).setBigUint64(0, sizeOverride ?? BigInt(json.byteLength), true);
  return new Blob([size, json, data]);
}
