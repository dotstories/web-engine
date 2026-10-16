import { getDevice } from "../../src/gpu/compute";
import { Graph } from "../../src/gpu/graph";
import { eulerStep, gaussian, sigmas } from "../../src/model/sampler";
import { LATENT_CHANNELS, TEXT_LAYERS, text, transformer } from "../../src/model/transformer";
import { loadTransformer } from "../../src/model/weights";
import { readHeader } from "../../src/weights/safetensors";

const $ = <T extends HTMLElement>(selector: string) => document.querySelector<T>(selector)!;
const seconds = (start: number) => +((performance.now() - start) / 1000).toFixed(1);
const TEXT_TOKENS = 64; // a typical prompt
const BLOCKS = 28;

$("#build").textContent = `Build ${__BUILD__}`;

$("#run").addEventListener("click", async () => {
  const status = $("#status");
  const model = $<HTMLInputElement>("#model").files?.[0];
  if (!model) {
    status.textContent = "Pick the model file first.";
    return;
  }
  const size = Number($<HTMLSelectElement>("#size").value);
  const latentSide = size / 8;
  const progress = $<HTMLProgressElement>("#progress");
  const run = $<HTMLButtonElement>("#run");
  run.disabled = true;
  progress.hidden = false;
  const graph = new Graph(await getDevice());
  try {
    let start = performance.now();
    const weights = await loadTransformer(graph, model, await readHeader(model), BLOCKS, (done, total) => {
      status.textContent = `Loading block ${done} of ${total}…`;
      progress.value = done / total;
    });
    await graph.device.queue.onSubmittedWorkDone();
    const loadTime = seconds(start);

    // Random stand-ins for the text encoder output and the starting noise (seeded, repeatable).
    const context = gaussian(TEXT_TOKENS * TEXT_LAYERS * 2560, 1);
    let latent = gaussian(LATENT_CHANNELS * latentSide * latentSide, 2);
    const schedule = sigmas();

    start = performance.now();
    const textCache = text(graph, weights, graph.buffer(context), TEXT_TOKENS).out;
    await graph.read(textCache, 4);
    const textTime = seconds(start);

    const stepTimes: number[] = [];
    for (let i = 0; i < schedule.length - 1; i++) {
      const stepStart = performance.now();
      const mark = graph.mark();
      const input = { latent, height: latentSide, width: latentSide, timestep: schedule[i], context, textTokens: TEXT_TOKENS, textCache };
      const { velocity } = await transformer(graph, weights, input, (b) => {
        status.textContent = `Step ${i + 1} of ${schedule.length - 1}, block ${b + 1} of ${BLOCKS}…`;
        progress.value = (i + (b + 1) / BLOCKS) / (schedule.length - 1);
      });
      graph.release(mark);
      latent = eulerStep(latent, velocity, schedule[i], schedule[i + 1]);
      stepTimes.push(seconds(stepStart));
    }
    const total = +(textTime + stepTimes.reduce((a, b) => a + b, 0)).toFixed(1);
    const finite = latent.every(Number.isFinite);

    status.textContent = finite ? `✅ Done: ${total} s for one ${size}×${size} image (8 steps).` : "❌ The result has NaN or infinite values.";
    const rows: [string, string][] = [
      ["Load 28 blocks", `${loadTime} s`],
      ["Text part (once)", `${textTime} s`],
      ["Per step", stepTimes.map((t) => `${t} s`).join(", ")],
      ["Total per image", `${total} s`],
    ];
    $("#results tbody").replaceChildren(
      ...rows.map(([label, value]) => {
        const row = document.createElement("tr");
        row.insertCell().textContent = label;
        row.insertCell().textContent = value;
        return row;
      }),
    );
    $("#results").hidden = false;
    const report = { build: __BUILD__, size, textTokens: TEXT_TOKENS, loadTime, textTime, stepTimes, total, finite, userAgent: navigator.userAgent };
    $("#report pre").textContent = JSON.stringify(report, null, 2);
    $("#report").hidden = false;
  } catch (e) {
    status.textContent = `❌ ${e instanceof Error ? e.message : String(e)}`;
  } finally {
    graph.destroy();
    run.disabled = false;
    progress.hidden = true;
  }
});

$("#copy").addEventListener("click", () => navigator.clipboard.writeText($("#report pre").textContent ?? ""));
