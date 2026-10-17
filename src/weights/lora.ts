/**
 * Finds the LoRA weights in a .safetensors file and matches them to the model's linear layers,
 * accepting the same key names ComfyUI does (comfy/lora.py, comfy/weight_adapter/lora.py):
 * ComfyUI (diffusion_model.*), kohya (lora_unet_*), lycoris_*, and diffusers/peft (transformer.*).
 *
 * The update for a weight W [n, k] is  W + strength · scale · up·down,  with down [rank, k],
 * up [n, rank] and scale = alpha / rank (1 without alpha).
 */
import { toFloat32 } from "./fp8";
import { readTensor, type Safetensors, type TensorInfo } from "./safetensors";

export type LoraTarget = { down: TensorInfo; up: TensorInfo; rank: number; scale: number };
/** targets: by model weight name without ".weight" (e.g. "blocks.0.attn.wq"). unused: LoRA keys that matched nothing. */
export type Lora = { targets: Map<string, LoraTarget>; unused: string[] };

export class LoraError extends Error {
  name = "LoraError";
}

/** comfy/utils.py krea2_to_diffusers, per block. */
const BLOCK_PARTS: Record<string, string[]> = {
  "attn.wq": ["attn.to_q"],
  "attn.wk": ["attn.to_k"],
  "attn.wv": ["attn.to_v"],
  "attn.gate": ["attn.to_gate"],
  "attn.wo": ["attn.to_out.0", "attn.to_out"],
  "mlp.gate": ["ff.gate"],
  "mlp.up": ["ff.up"],
  "mlp.down": ["ff.down"],
};
const BLOCKS: [RegExp, string][] = [
  [/^blocks\.(\d+)\.(.+)$/, "transformer_blocks"],
  [/^txtfusion\.layerwise_blocks\.(\d+)\.(.+)$/, "text_fusion.layerwise_blocks"],
  [/^txtfusion\.refiner_blocks\.(\d+)\.(.+)$/, "text_fusion.refiner_blocks"],
];
const SHARED: Record<string, string> = {
  first: "img_in",
  "tmlp.0": "time_embed.linear_1",
  "tmlp.2": "time_embed.linear_2",
  "tproj.1": "time_mod_proj",
  "txtmlp.1": "txt_in.linear_1",
  "txtmlp.3": "txt_in.linear_2",
  "txtfusion.projector": "text_fusion.projector",
  "last.linear": "final_layer.linear",
};

/** The diffusers name(s) of a ComfyUI Krea 2 weight (without ".weight"). */
export function diffusersName(name: string): string[] {
  if (name in SHARED) return [SHARED[name]];
  for (const [pattern, prefix] of BLOCKS) {
    const m = pattern.exec(name);
    if (m) return (BLOCK_PARTS[m[2]] ?? []).map((part) => `${prefix}.${m[1]}.${part}`);
  }
  return [];
}

/** Every LoRA prefix that points at this model weight. */
function prefixes(name: string): string[] {
  const underscored = (s: string) => s.replace(/\./g, "_");
  const comfy = [name, `diffusion_model.${name}`, `lora_unet_${underscored(name)}`];
  const diffusers = diffusersName(name).flatMap((d) => [d, `diffusion_model.${d}`, `transformer.${d}`, `lycoris_${underscored(d)}`]);
  return [...comfy, ...diffusers];
}

/** [up, down] key pairs ComfyUI's LoRAAdapter.load checks, in its order. */
const PAIRS = (x: string): [string, string][] => [
  [`${x}.lora_up.weight`, `${x}.lora_down.weight`],
  [`${x}_lora.up.weight`, `${x}_lora.down.weight`],
  [`${x}.lora_B.weight`, `${x}.lora_A.weight`],
  [`${x}.lora.up.weight`, `${x}.lora.down.weight`],
  [`${x}.lora_B`, `${x}.lora_A`],
  [`${x}.lora_linear_layer.up.weight`, `${x}.lora_linear_layer.down.weight`],
  [`${x}.lora_B.default.weight`, `${x}.lora_A.default.weight`],
];

const fits = (down: number[], up: number[], weight: number[]) =>
  down.length === 2 && up.length === 2 && down[0] === up[1] && down[1] === weight[1] && up[0] === weight[0];

/** Matches the LoRA file to the model. Reads only the header and the small alpha scalars. */
export async function matchLora(file: Blob, lora: Safetensors, model: Safetensors): Promise<Lora> {
  const used = new Set<string>();
  const targets = new Map<string, LoraTarget>();

  for (const info of model.tensors.values()) {
    if (!info.name.endsWith(".weight") || info.shape.length !== 2) continue;
    const name = info.name.slice(0, -".weight".length);
    for (const x of prefixes(name)) {
      const pair = PAIRS(x).find(([u, d]) => lora.tensors.has(u) && lora.tensors.has(d));
      if (!pair) continue;
      const [up, down] = pair.map((key) => lora.tensors.get(key)!);
      if (lora.tensors.has(`${x}.dora_scale`) || lora.tensors.has(`${x}.lora_mid.weight`)) {
        throw new LoraError(`${name}: DoRA / mid LoRA weights are not supported yet`);
      }
      if (!fits(down.shape, up.shape, info.shape)) {
        const s = (shape: number[]) => JSON.stringify(shape);
        throw new LoraError(`${name}: LoRA down ${s(down.shape)} and up ${s(up.shape)} do not fit the weight ${s(info.shape)}`);
      }
      const rank = down.shape[0];
      const alphaInfo = lora.tensors.get(`${x}.alpha`);
      const alpha = alphaInfo ? toFloat32(alphaInfo.dtype, await readTensor(file, alphaInfo))[0] : undefined;
      targets.set(name, { down, up, rank, scale: alpha === undefined ? 1 : alpha / rank });
      for (const key of [up.name, down.name, alphaInfo?.name]) if (key) used.add(key);
      break;
    }
  }

  if (targets.size === 0) throw new LoraError("No LoRA weights for this model in the file");
  const unused = [...lora.tensors.keys()].filter((key) => !used.has(key)).sort();
  return { targets, unused };
}

/**
 * One LoRA matrix ready for the GPU: down [rank, k] (pad "rows") or up [n, rank] (pad "cols").
 * The matmul needs rank to be a multiple of 4, so other ranks get zero rows / columns (as f32).
 * bf16 stays bf16 (half the memory); F32 and F16 become f32.
 */
export function loraMatrix(dtype: string, bytes: Uint8Array, rows: number, cols: number, pad: "rows" | "cols") {
  const rank = pad === "rows" ? rows : cols;
  const padded = Math.ceil(rank / 4) * 4;
  if (dtype === "BF16" && padded === rank) {
    return { dtype: "bf16" as const, data: new Uint32Array(bytes.slice().buffer), rows, cols };
  }
  const values = toFloat32(dtype, bytes);
  if (padded === rank) return { dtype: "f32" as const, data: values, rows, cols };
  const [r, c] = pad === "rows" ? [padded, cols] : [rows, padded];
  const out = new Float32Array(r * c);
  for (let i = 0; i < rows; i++) out.set(values.subarray(i * cols, (i + 1) * cols), i * c);
  return { dtype: "f32" as const, data: out, rows: r, cols: c };
}

/** W [n, k] += scale · up [n, rank] · down [rank, k], in place (for small f32 weights merged on the CPU). */
export function mergeLora<T extends Float32Array>(weight: T, n: number, k: number, down: Float32Array, up: Float32Array, rank: number, scale: number): T {
  for (let i = 0; i < n; i++)
    for (let j = 0; j < k; j++) {
      let sum = 0;
      for (let r = 0; r < rank; r++) sum += up[i * rank + r] * down[r * k + j];
      weight[i * k + j] += scale * sum;
    }
  return weight;
}

/** What a matched LoRA holds, for the page: layers, dtypes and ranks in the file, bytes of its matrices, unused keys. */
export function describeLora(lora: Lora) {
  const targets = [...lora.targets.values()];
  const unique = <T>(values: T[]) => [...new Set(values)].sort();
  return {
    layers: targets.length,
    dtypes: unique(targets.flatMap((t) => [t.down.dtype, t.up.dtype])),
    ranks: unique(targets.map((t) => t.rank)).sort((a, b) => a - b),
    bytes: targets.reduce((sum, t) => sum + t.down.size + t.up.size, 0),
    unused: lora.unused.length,
  };
}
