import { getDevice } from "../gpu/compute";
import { Graph } from "../gpu/graph";
import { eulerStep, gaussian, sigmas } from "../model/sampler";
import { encode, loadTextEncoder } from "../model/text-encoder";
import { LATENT_CHANNELS, text, transformer } from "../model/transformer";
import { decode, loadVae, toRgba, vaeInput } from "../model/vae";
import { loadTransformer } from "../model/weights";
import { kreaPrompt } from "../text/prompt";
import type { Qwen2Tokenizer } from "../text/tokenizer";
import { matchLora } from "../weights/lora";
import { readHeader } from "../weights/safetensors";
import type { ModelFiles } from "./files";

/** loraStrength: ComfyUI's strength_model for files.lora (0 or no file = no LoRA). */
export type Request = { prompt: string; width: number; height: number; seed: number; loraStrength?: number };
export type Progress = { stage: "text" | "steps" | "decode"; message: string; fraction: number };

/**
 * Prompt → image with Krea 2 Turbo, all on this computer's GPU. Fits a 16 GB card by running one model
 * at a time: text encoder (3.5 GB) → free → transformer (13 GB, 8 steps) → free → image decoder.
 */
export async function generate(files: ModelFiles, tokenizer: Qwen2Tokenizer, req: Request, onProgress: (p: Progress) => void) {
  const device = await getDevice();
  const [lh, lw] = [req.height / 8, req.width / 8];
  const times: Record<string, number> = {};
  const timed = async <T>(name: string, f: () => Promise<T>) => {
    const start = performance.now();
    const out = await f();
    times[name] = +((performance.now() - start) / 1000).toFixed(1);
    return out;
  };

  // 1. Text encoder
  const { cond, kept } = await timed("text", async () => {
    const graph = new Graph(device);
    try {
      const ids = tokenizer.encode(kreaPrompt(req.prompt));
      const w = await loadTextEncoder(graph, files.textEncoder, await readHeader(files.textEncoder), (d, t) =>
        onProgress({ stage: "text", message: `Loading the text encoder (${d}/${t})…`, fraction: (0.1 * d) / t }),
      );
      onProgress({ stage: "text", message: "Reading the prompt…", fraction: 0.1 });
      return await encode(graph, w, ids);
    } finally {
      graph.destroy();
    }
  });

  // 2. Transformer: 8 Turbo steps
  let latent = gaussian(LATENT_CHANNELS * lh * lw, req.seed);
  let loraLayers = 0;
  await timed("steps", async () => {
    const graph = new Graph(device);
    try {
      const st = await readHeader(files.transformer);
      const strength = req.loraStrength ?? 0;
      const lora = files.lora && strength ? { file: files.lora, lora: await matchLora(files.lora, await readHeader(files.lora), st), strength } : undefined;
      loraLayers = lora?.lora.targets.size ?? 0;
      const w = await loadTransformer(
        graph,
        files.transformer,
        st,
        28,
        (d, t) => onProgress({ stage: "steps", message: `Loading the image model (${d}/${t})…`, fraction: 0.1 + (0.2 * d) / t }),
        lora,
      );
      const textCache = text(graph, w, graph.buffer(cond), kept).out;
      const schedule = sigmas();
      for (let i = 0; i < schedule.length - 1; i++) {
        const mark = graph.mark();
        const input = { latent, height: lh, width: lw, timestep: schedule[i], context: cond, textTokens: kept, textCache };
        const { velocity } = await transformer(graph, w, input, (b) =>
          onProgress({ stage: "steps", message: `Step ${i + 1} of ${schedule.length - 1}…`, fraction: 0.3 + (0.6 * (i + (b + 1) / 28)) / (schedule.length - 1) }),
        );
        graph.release(mark);
        latent = eulerStep(latent, velocity, schedule[i], schedule[i + 1]);
      }
    } finally {
      graph.destroy();
    }
  });

  // 3. Image decoder
  const rgba = await timed("decode", async () => {
    const graph = new Graph(device);
    try {
      onProgress({ stage: "decode", message: "Painting the image…", fraction: 0.92 });
      const w = await loadVae(graph, files.vae, await readHeader(files.vae));
      const z = graph.buffer(vaeInput(latent, lh * lw));
      const { image, height, width } = await decode(graph, w, z, lh, lw);
      return toRgba(new Float32Array(await graph.read(image, height * width * 3 * 4)), height * width);
    } finally {
      graph.destroy();
    }
  });

  onProgress({ stage: "decode", message: "Done", fraction: 1 });
  return { rgba, width: req.width, height: req.height, times, loraLayers };
}
