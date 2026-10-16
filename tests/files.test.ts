import { describe, expect, it } from "vitest";
import { findModelFiles, type Folder } from "../src/pipeline/files";

/** A fake folder tree: { name: subtree } for folders, { name: true } for files. */
type Tree = { [name: string]: Tree | true };
function folder(name: string, tree: Tree): Folder {
  return {
    name,
    getDirectoryHandle: async (n) => {
      const sub = tree[n];
      if (!sub || sub === true) throw new DOMException("not found", "NotFoundError");
      return folder(n, sub);
    },
    getFileHandle: async (n) => {
      if (tree[n] !== true) throw new DOMException("not found", "NotFoundError");
      return { getFile: async () => new File(["x"], n) };
    },
  };
}

const models: Tree = {
  diffusion_models: { "krea2_turbo_fp8_scaled.safetensors": true },
  text_encoders: { "qwen3vl_4b_fp8_scaled.safetensors": true },
  vae: { "qwen_image_vae.safetensors": true },
};

describe("findModelFiles", () => {
  it("finds the three Krea 2 files in a ComfyUI models folder", async () => {
    const { files, missing } = await findModelFiles(folder("models", models));
    expect(missing).toEqual([]);
    expect([files.transformer?.name, files.textEncoder?.name, files.vae?.name]).toEqual([
      "krea2_turbo_fp8_scaled.safetensors",
      "qwen3vl_4b_fp8_scaled.safetensors",
      "qwen_image_vae.safetensors",
    ]);
  });

  it("also accepts the folder that contains 'models' (ComfyUI, Comfy Desktop's ComfyUI-Shared)", async () => {
    const { missing } = await findModelFiles(folder("ComfyUI-Shared", { models }));
    expect(missing).toEqual([]);
  });

  it("says which files are missing", async () => {
    const { files, missing } = await findModelFiles(folder("models", { vae: { "qwen_image_vae.safetensors": true } }));
    expect(files.vae).toBeDefined();
    expect(missing).toEqual(["diffusion_models/krea2_turbo_fp8_scaled.safetensors", "text_encoders/qwen3vl_4b_fp8_scaled.safetensors"]);
  });

  it("finds the optional Dark Fantasy Kit LoRA in loras/, and never lists it as missing", async () => {
    const withLora = await findModelFiles(folder("models", { ...models, loras: { "krea2_darkbrush.safetensors": true } }));
    expect(withLora.files.lora?.name).toBe("krea2_darkbrush.safetensors");
    const without = await findModelFiles(folder("models", models));
    expect(without.files.lora).toBeUndefined();
    expect(without.missing).toEqual([]);
  });
});
