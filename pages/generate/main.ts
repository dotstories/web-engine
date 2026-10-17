import { findModelFiles, type Folder, type ModelFiles } from "../../src/pipeline/files";
import { generate } from "../../src/pipeline/generate";
import { IMAGE_SIZES, parseSize, sizeValue } from "../../src/pipeline/sizes";
import { Qwen2Tokenizer } from "../../src/text/tokenizer";

const $ = <T extends HTMLElement>(selector: string) => document.querySelector<T>(selector)!;
const tokenizerFile = (name: string) => new URL(`../../tokenizers/qwen2/${name}`, location.href);
const picker = (window as unknown as { showDirectoryPicker?: (options?: object) => Promise<Folder> }).showDirectoryPicker;

let files: ModelFiles | null = null;
$("#build").textContent = `Build ${__BUILD__}`;
for (const s of IMAGE_SIZES) {
  const value = sizeValue(s);
  $<HTMLSelectElement>("#size").add(new Option(`${s.width} × ${s.height} (${s.note})`, value, false, value === "1024x1024"));
}

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
    const gb = (f: File) => (f.size / 1024 ** 3).toFixed(1);
    status.textContent = `✅ Found all 3 files (${gb(files.transformer)} + ${gb(files.textEncoder)} + ${gb(files.vae)} GB).`;
    const lora = $<HTMLSelectElement>("#lora");
    lora.querySelector("option[value=darkbrush]")?.remove();
    if (files.lora) {
      lora.add(new Option(`${files.lora.name} (Dark Fantasy Kit)`, "darkbrush", true, true));
      status.textContent += ` LoRA found: ${files.lora.name}.`;
    }
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") return; // closed the picker
    status.textContent = `❌ ${e instanceof Error ? e.message : String(e)}`;
  }
});

$("#generate").addEventListener("click", async () => {
  const status = $("#status");
  if (!files) {
    status.textContent = "Choose your ComfyUI models folder first.";
    return;
  }
  if (!$<HTMLInputElement>("#license").checked) {
    status.textContent = "Please accept the Krea 2 Community License first.";
    return;
  }
  const { width, height } = parseSize($<HTMLSelectElement>("#size").value);
  const strength = $<HTMLSelectElement>("#lora").value ? Number($<HTMLInputElement>("#strength").value) : 0;
  const request = {
    prompt: $<HTMLTextAreaElement>("#prompt").value.trim(),
    width,
    height,
    seed: Number($<HTMLInputElement>("#seed").value) >>> 0,
    loraStrength: Number.isFinite(strength) ? strength : 0,
  };
  if (!request.prompt) {
    status.textContent = "Write a prompt first.";
    return;
  }
  const button = $<HTMLButtonElement>("#generate");
  const progress = $<HTMLProgressElement>("#progress");
  button.disabled = true;
  progress.hidden = false;
  const start = performance.now();
  try {
    const tokenizer = await Qwen2Tokenizer.load(tokenizerFile("vocab.json"), tokenizerFile("merges.txt"));
    const { rgba, width, height, times, stepTimes, lora } = await generate(files, tokenizer, request, (p) => {
      status.textContent = p.message;
      progress.value = p.fraction;
    });
    const canvas = $<HTMLCanvasElement>("#image");
    [canvas.width, canvas.height] = [width, height];
    canvas.getContext("2d")!.putImageData(new ImageData(rgba, width, height), 0, 0);
    canvas.hidden = false;
    const link = $<HTMLAnchorElement>("#download");
    canvas.toBlob((blob) => {
      if (blob) link.href = URL.createObjectURL(blob);
      link.hidden = false;
    }, "image/png");
    const total = ((performance.now() - start) / 1000).toFixed(0);
    status.textContent = `✅ Done in ${total} s.`;
    const mb = (bytes: number) => Math.round(bytes / 1024 ** 2);
    const loraText = lora
      ? ` · LoRA ${request.loraStrength} on ${lora.layers} layers (${lora.dtypes.join("/")}, rank ${lora.ranks.join("/")}, ${mb(lora.bytes)} MB, ${lora.unused} unused keys)`
      : "";
    $("#times").textContent =
      `text ${times.text} s · load ${times.load} s · 8 steps ${times.steps} s (${stepTimes.join(", ")}) · decode ${times.decode} s${loraText} · build ${__BUILD__}`;
  } catch (e) {
    status.textContent = `❌ ${e instanceof Error ? e.message : String(e)}`;
  } finally {
    button.disabled = false;
    progress.hidden = true;
  }
});
