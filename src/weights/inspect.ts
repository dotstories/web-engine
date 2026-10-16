import type { Safetensors, TensorInfo } from "./safetensors";

/** Tensors only Krea 2 has (ComfyUI files add the "model.diffusion_model." prefix). */
const KREA_KEYS = ["txtfusion.projector.weight", "model.diffusion_model.txtfusion.projector.weight"];
const SCALE_SUFFIXES = [".scale_weight", ".weight_scale"];

export type Fp8Weight = { weight: TensorInfo; scale: TensorInfo | null };
export type BlockGroup = { prefix: string; count: number };

/** fp8 weights with their per-layer scale (ComfyUI "fp8_scaled": real weight = fp8 × scale). */
export function fp8Weights(st: Safetensors): Fp8Weight[] {
  const pairs: Fp8Weight[] = [];
  for (const t of st.tensors.values()) {
    if (!t.dtype.startsWith("F8_") || !t.name.endsWith(".weight")) continue;
    const base = t.name.slice(0, -".weight".length);
    const scale = SCALE_SUFFIXES.map((s) => st.tensors.get(base + s)).find(Boolean) ?? null;
    pairs.push({ weight: t, scale });
  }
  return pairs.sort((a, b) => a.weight.name.localeCompare(b.weight.name));
}

/** Repeated blocks ("x.blocks.0.…", "x.blocks.1.…") and how many of each, biggest first. */
export function blockGroups(names: Iterable<string>): BlockGroup[] {
  const groups = new Map<string, Set<string>>();
  for (const name of names) {
    const m = /^(.*?)\.(\d+)\./.exec(name);
    if (!m) continue;
    if (!groups.has(m[1])) groups.set(m[1], new Set());
    groups.get(m[1])!.add(m[2]);
  }
  return [...groups].map(([prefix, ids]) => ({ prefix, count: ids.size })).sort((a, b) => b.count - a.count);
}

export type Summary = {
  tensors: number;
  bytes: number;
  isKrea: boolean;
  dtypes: Record<string, number>;
  fp8: { weights: number; scaled: number };
  blocks: BlockGroup[];
  firstBlock: { name: string; dtype: string; shape: number[] }[];
};

export function summarize(st: Safetensors): Summary {
  const all = [...st.tensors.values()];
  const dtypes: Record<string, number> = {};
  for (const t of all) dtypes[t.dtype] = (dtypes[t.dtype] ?? 0) + 1;
  const fp8 = fp8Weights(st);
  const blocks = blockGroups(st.tensors.keys());
  const first = blocks[0] ? `${blocks[0].prefix}.0.` : null;
  return {
    tensors: all.length,
    bytes: all.reduce((sum, t) => sum + t.size, 0),
    isKrea: KREA_KEYS.some((k) => st.tensors.has(k)),
    dtypes,
    fp8: { weights: fp8.length, scaled: fp8.filter((p) => p.scale).length },
    blocks,
    firstBlock: first
      ? all
          .filter((t) => t.name.startsWith(first))
          .map(({ name, dtype, shape }) => ({ name, dtype, shape }))
          .sort((a, b) => a.name.localeCompare(b.name))
      : [],
  };
}
