/** Where Krea 2's three files live in a ComfyUI models folder (and their usual names). */
export const KREA2_FILES = {
  transformer: { folder: "diffusion_models", names: ["krea2_turbo_fp8_scaled.safetensors"] },
  textEncoder: { folder: "text_encoders", names: ["qwen3vl_4b_fp8_scaled.safetensors"] },
  vae: { folder: "vae", names: ["qwen_image_vae.safetensors"] },
} as const;

/** Optional: the Dark Fantasy Kit's LoRA (Comfy-Org/Krea-2 loras/), used at strength 0.8 there. */
export const KREA2_LORA = { folder: "loras", name: "krea2_darkbrush.safetensors" } as const;

export type ModelFiles = { transformer: File; textEncoder: File; vae: File; lora?: File };
export type Part = keyof typeof KREA2_FILES;

/** A folder handle as far as we need it (File System Access API), so tests can pass a fake. */
export type Folder = {
  name: string;
  getDirectoryHandle(name: string): Promise<Folder>;
  getFileHandle(name: string): Promise<{ getFile(): Promise<File> }>;
};

async function tryFolder(folder: Folder, name: string): Promise<Folder | null> {
  return folder.getDirectoryHandle(name).catch(() => null);
}

/**
 * Finds the three files in the picked folder: a ComfyUI "models" folder, or a folder that contains one
 * (e.g. ComfyUI itself, or Comfy Desktop's "ComfyUI-Shared"). Returns what it found and what is missing.
 * The LoRA is optional: found when present, never "missing".
 */
export async function findModelFiles(picked: Folder): Promise<{ files: Partial<ModelFiles>; missing: string[] }> {
  const models = picked.name === "models" ? picked : ((await tryFolder(picked, "models")) ?? picked);
  const files: Partial<ModelFiles> = {};
  const missing: string[] = [];
  for (const [part, { folder, names }] of Object.entries(KREA2_FILES) as [Part, (typeof KREA2_FILES)[Part]][]) {
    const dir = await tryFolder(models, folder);
    let found: File | null = null;
    for (const name of names) {
      found = dir ? await dir.getFileHandle(name).then((h) => h.getFile()).catch(() => null) : null;
      if (found) break;
    }
    if (found) files[part] = found;
    else missing.push(`${folder}/${names[0]}`);
  }
  const loras = await tryFolder(models, KREA2_LORA.folder);
  const lora = loras ? await loras.getFileHandle(KREA2_LORA.name).then((h) => h.getFile()).catch(() => null) : null;
  if (lora) files.lora = lora;
  return { files, missing };
}
