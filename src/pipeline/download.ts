import { KREA2_FILES, KREA2_LORA, type Part } from "./files";

/** Where the Krea 2 files come from: Comfy-Org's Hugging Face repo (no sign-in), same names as ComfyUI uses. */
export const KREA2_SOURCE = "https://huggingface.co/Comfy-Org/Krea-2/resolve/main";

export type Download = { folder: string; name: string; size: number };

/** The files to download, with their exact sizes (bytes) on Hugging Face. */
export const KREA2_DOWNLOADS: Record<Part, Download> = {
  transformer: { folder: KREA2_FILES.transformer.folder, name: KREA2_FILES.transformer.names[0], size: 13141730784 },
  textEncoder: { folder: KREA2_FILES.textEncoder.folder, name: KREA2_FILES.textEncoder.names[0], size: 5242467968 },
  vae: { folder: KREA2_FILES.vae.folder, name: KREA2_FILES.vae.names[0], size: 253806246 },
};
export const KREA2_LORA_DOWNLOAD: Download = { folder: KREA2_LORA.folder, name: KREA2_LORA.name, size: 469291992 };

/** The downloads for findModelFiles()'s "missing" list ("folder/name"). */
export function downloadsFor(missing: string[]): Download[] {
  return Object.values(KREA2_DOWNLOADS).filter((d) => missing.includes(`${d.folder}/${d.name}`));
}

/** The File System Access API as far as downloading needs it (a folder picked with mode "readwrite"). */
export type Writable = {
  seek(position: number): Promise<void>;
  truncate(size: number): Promise<void>;
  write(data: Uint8Array): Promise<void>;
  close(): Promise<void>;
};
export type WritableFile = {
  getFile(): Promise<File>;
  createWritable(options?: { keepExistingData?: boolean }): Promise<Writable>;
  move?(name: string): Promise<void>;
};
export type WritableFolder = {
  getDirectoryHandle(name: string, options?: { create?: boolean }): Promise<WritableFolder>;
  getFileHandle(name: string, options?: { create?: boolean }): Promise<WritableFile>;
};

export type DownloadOptions = { signal?: AbortSignal; fetch?: typeof fetch; source?: string };

/**
 * Downloads one file into the models folder (e.g. models/vae/qwen_image_vae.safetensors), the ComfyUI layout,
 * so ComfyUI finds it too. Writes "<name>.part" and renames it when complete, so a half file is never used.
 * Stopped or failed downloads keep what they got: the next call continues from there (HTTP Range).
 */
export async function downloadFile(
  models: WritableFolder,
  d: Download,
  onProgress: (done: number, total: number) => void,
  { signal, fetch = globalThis.fetch, source = KREA2_SOURCE }: DownloadOptions = {},
): Promise<void> {
  const dir = await models.getDirectoryHandle(d.folder, { create: true });
  const part = await dir.getFileHandle(`${d.name}.part`, { create: true });
  if (!part.move) throw new Error("This browser can't save downloads into a folder. Update Chrome or Edge.");
  let done = (await part.getFile()).size;
  if (done > d.size) done = 0; // not ours: start over
  onProgress(done, d.size);
  if (done < d.size) {
    const writer = await part.createWritable({ keepExistingData: done > 0 });
    try {
      const res = await fetch(`${source}/${d.folder}/${d.name}`, { headers: done ? { Range: `bytes=${done}-` } : {}, signal });
      if (!res.ok || !res.body) throw new Error(`Download of ${d.name} failed (HTTP ${res.status})`);
      if (done && res.status !== 206) done = 0; // the server sent the whole file
      await writer.truncate(done);
      await writer.seek(done);
      const reader = res.body.getReader();
      for (;;) {
        const chunk = await reader.read();
        if (chunk.done) break;
        await writer.write(chunk.value);
        done += chunk.value.byteLength;
        onProgress(done, d.size);
      }
    } finally {
      await writer.close(); // keeps what arrived, also after a cancel or a dropped connection
    }
  }
  const size = (await part.getFile()).size;
  if (size !== d.size) throw new Error(`${d.name} is incomplete (${size} of ${d.size} bytes). Try again to continue.`);
  await part.move(d.name);
}

/** Downloads several files one after another, with one overall progress. */
export async function downloadModels(
  models: WritableFolder,
  downloads: Download[],
  onProgress: (done: number, total: number, file: string) => void,
  options: DownloadOptions = {},
): Promise<void> {
  const total = downloads.reduce((s, d) => s + d.size, 0);
  let before = 0;
  for (const d of downloads) {
    await downloadFile(models, d, (done) => onProgress(before + done, total, d.name), options);
    before += d.size;
  }
}
