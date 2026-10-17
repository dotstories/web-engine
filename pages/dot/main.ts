import { psnr, ssim } from "../../src/check/similarity";
import { findModelFiles, type Folder, type ModelFiles } from "../../src/pipeline/files";
import { generate } from "../../src/pipeline/generate";
import { Qwen2Tokenizer } from "../../src/text/tokenizer";
import { WEEPING_GATE as DOT } from "./recipe";

const $ = <T extends HTMLElement>(selector: string) => document.querySelector<T>(selector)!;
const url = (path: string) => new URL(path, location.href);
const nativePicker = (window as unknown as { showDirectoryPicker?: (options?: object) => Promise<Folder> }).showDirectoryPicker;
/** `?models=browser` (for testing): read a copy of the models from this site's private browser storage, no folder prompt. */
const browserModels = async () => (await navigator.storage.getDirectory()).getDirectoryHandle("models") as Promise<unknown> as Promise<Folder>;
const picker = new URLSearchParams(location.search).get("models") === "browser" ? browserModels : nativePicker && ((options?: object) => nativePicker(options));

let files: ModelFiles | null = null;
$("#build").textContent = `Build ${__BUILD__}`;
$("#recipe").textContent = `Krea 2 Turbo · ${DOT.width} × ${DOT.height} · seed ${DOT.seed} · 8 steps · no LoRA`;
$<HTMLImageElement>("#reference").src = url(DOT.reference).href;
const prompt = fetch(url(DOT.prompt)).then((r) => {
  if (!r.ok) throw new Error(`Could not load the prompt (${r.status})`);
  return r.text();
});
prompt.then((text) => ($("#prompt").textContent = text)).catch((e) => ($("#prompt").textContent = String(e)));

$("#pick").addEventListener("click", async () => {
  const status = $("#files");
  if (!picker) {
    status.textContent = "This browser can't open folders. Use Chrome or Edge on a desktop.";
    return;
  }
  try {
    const found = await findModelFiles(await picker({ id: "comfyui-models", mode: "read" }));
    if (found.missing.length) {
      status.textContent = `Missing: ${found.missing.join(", ")}`;
      return;
    }
    files = found.files as ModelFiles;
    status.textContent = "✅ Found all 3 files.";
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") return; // closed the picker
    status.textContent = `❌ ${e instanceof Error ? e.message : String(e)}`;
  }
});

/** The reference as RGBA bytes, at its own size. */
async function referencePixels(): Promise<ImageData> {
  const bitmap = await createImageBitmap(await (await fetch(url(DOT.reference))).blob());
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(bitmap, 0, 0);
  return ctx.getImageData(0, 0, bitmap.width, bitmap.height);
}

$("#make").addEventListener("click", async () => {
  const status = $("#status");
  if (!files) {
    status.textContent = "Choose your ComfyUI models folder first.";
    return;
  }
  if (!$<HTMLInputElement>("#license").checked) {
    status.textContent = "Please accept the Krea 2 Community License first.";
    return;
  }
  const [make, cancel, progress] = [$<HTMLButtonElement>("#make"), $<HTMLButtonElement>("#cancel"), $<HTMLProgressElement>("#progress")];
  const controller = new AbortController();
  cancel.onclick = () => {
    controller.abort();
    status.textContent = "Cancelling…";
  };
  make.disabled = true;
  cancel.hidden = false;
  progress.hidden = false;
  $("#score").textContent = "";
  const start = performance.now();
  try {
    const tokenizer = await Qwen2Tokenizer.load(url("../../tokenizers/qwen2/vocab.json"), url("../../tokenizers/qwen2/merges.txt"));
    const request = { prompt: await prompt, width: DOT.width, height: DOT.height, seed: DOT.seed, loraStrength: DOT.loraStrength };
    const { rgba, width, height, times, stepTimes } = await generate(files, tokenizer, request, (p) => {
      status.textContent = p.message;
      progress.value = p.fraction;
    }, { signal: controller.signal });
    const canvas = $<HTMLCanvasElement>("#image");
    [canvas.width, canvas.height] = [width, height];
    canvas.getContext("2d")!.putImageData(new ImageData(rgba, width, height), 0, 0);
    canvas.hidden = false;
    $("#mine").textContent = "Your browser";
    canvas.toBlob((blob) => {
      const link = $<HTMLAnchorElement>("#download");
      if (blob) link.href = URL.createObjectURL(blob);
      link.hidden = false;
    }, "image/png");
    status.textContent = `✅ Done in ${((performance.now() - start) / 1000).toFixed(0)} s.`;
    $("#times").textContent = `text ${times.text} s · load ${times.load} s · 8 steps ${times.steps} s (${stepTimes.join(", ")}) · decode ${times.decode} s · build ${__BUILD__}`;

    const ref = await referencePixels();
    if (ref.width !== width || ref.height !== height) throw new Error(`The reference is ${ref.width} × ${ref.height}, the image ${width} × ${height}`);
    const s = ssim(rgba, ref.data, width, height);
    const p = psnr(rgba, ref.data);
    $("#score").textContent = `Similarity to ComfyUI: SSIM ${s.toFixed(3)} · PSNR ${p.toFixed(1)} dB (1.000 / ∞ = identical)`;
  } catch (e) {
    status.textContent = controller.signal.aborted ? "Cancelled." : `❌ ${e instanceof Error ? e.message : String(e)}`;
  } finally {
    make.disabled = false;
    cancel.hidden = true;
    progress.hidden = true;
  }
});
