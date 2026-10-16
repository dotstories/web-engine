import { getDevice } from "../../src/gpu/compute";
import { Graph } from "../../src/gpu/graph";
import { compare } from "../../src/model/compare";
import { transformer } from "../../src/model/transformer";
import { loadFloats, loadTransformer } from "../../src/model/weights";
import { readHeader } from "../../src/weights/safetensors";

const $ = <T extends HTMLElement>(selector: string) => document.querySelector<T>(selector)!;
const file = (id: string) => $<HTMLInputElement>(id).files?.[0];
const seconds = (start: number) => +((performance.now() - start) / 1000).toFixed(2);
const PASS = 0.999;

$("#build").textContent = `Build ${__BUILD__}`;

$("#run").addEventListener("click", async () => {
  const status = $("#status");
  const [model, reference] = [file("#model"), file("#reference")];
  if (!model || !reference) {
    status.textContent = "Pick the model file and the reference file first.";
    return;
  }
  const run = $<HTMLButtonElement>("#run");
  run.disabled = true;
  const graph = new Graph(await getDevice());
  try {
    status.textContent = "Reading the reference…";
    const ref = await readHeader(reference);
    const meta = { blocks: Number(ref.metadata.blocks), text: Number(ref.metadata.text), latent: Number(ref.metadata.latent) };
    const floats = (name: string) => loadFloats(reference, ref, name);
    const [latent, context, timestep] = [await floats("x"), await floats("context"), (await floats("timestep"))[0]];

    let start = performance.now();
    const weights = await loadTransformer(graph, model, await readHeader(model), meta.blocks, (done, total) => {
      status.textContent = `Loading block ${done} of ${total}…`;
    });
    const loadTime = seconds(start);

    status.textContent = "Running…";
    start = performance.now();
    const input = { latent, height: meta.latent, width: meta.latent, timestep, context, textTokens: meta.text };
    const { velocity, steps } = await transformer(graph, weights, input);
    const runTime = seconds(start);

    const results = [];
    for (const [name, buffer] of Object.entries(steps)) {
      const expected = await floats(`step.${name}`);
      results.push({ step: name, ...compare(new Float32Array(await graph.read(buffer, expected.byteLength)), expected) });
    }
    results.push({ step: "output", ...compare(velocity, await floats("out")) });
    const pass = results.every((r) => r.cosine >= PASS);

    status.textContent = pass ? "✅ Every step matches ComfyUI." : "❌ Some steps don't match ComfyUI.";
    $("#results tbody").replaceChildren(
      ...results.map((r) => {
        const row = document.createElement("tr");
        for (const text of [r.step, r.cosine.toFixed(6), r.relativeError.toExponential(2), r.cosine >= PASS ? "✅" : "❌"]) {
          row.insertCell().textContent = text;
        }
        return row;
      }),
    );
    $("#results").hidden = false;
    $("#report pre").textContent = JSON.stringify({ build: __BUILD__, ...meta, timestep, pass, loadTime, runTime, results }, null, 2);
    $("#report").hidden = false;
  } catch (e) {
    status.textContent = `❌ ${e instanceof Error ? e.message : String(e)}`;
  } finally {
    graph.destroy();
    run.disabled = false;
  }
});

$("#copy").addEventListener("click", () => navigator.clipboard.writeText($("#report pre").textContent ?? ""));
