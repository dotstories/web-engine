export type Size = { width: number; height: number };

/**
 * ComfyUI's ResolutionSelector: about `megapixels` (binary: 1 MP = 1024²) at the aspect ratio,
 * each side rounded to the nearest `multiple`. The Dark Fantasy Kit uses 16:9, 2 MP, 8 → 1928 × 1088.
 */
export function resolution([w, h]: [number, number], megapixels: number, multiple = 8): Size {
  const scale = Math.sqrt((megapixels * 1024 * 1024) / (w * h));
  const snap = (x: number) => Math.round(x / multiple) * multiple;
  return { width: snap(w * scale), height: snap(h * scale) };
}

/** Sizes offered on the generate page. */
export const IMAGE_SIZES: (Size & { note: string })[] = [
  { width: 512, height: 512, note: "fast" },
  { width: 1024, height: 1024, note: "1 MP" },
  { ...resolution([16, 9], 1), note: "16:9, 1 MP" },
  { ...resolution([16, 9], 2), note: "16:9, 2 MP, Dark Fantasy Kit" },
];

export const sizeValue = ({ width, height }: Size) => `${width}x${height}`;

/** "1928x1088" → { width: 1928, height: 1088 }. Both sides must be multiples of 8 (the VAE's factor). */
export function parseSize(value: string): Size {
  const match = /^(\d+)x(\d+)$/.exec(value);
  if (!match) throw new Error(`Not a size: ${value}`);
  const [width, height] = [Number(match[1]), Number(match[2])];
  if (width % 8 || height % 8 || !width || !height) throw new Error(`Width and height must be a multiple of 8 (got ${width} × ${height})`);
  return { width, height };
}
