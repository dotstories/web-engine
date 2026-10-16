import { getDevice } from "../../src/gpu/compute";
import { Graph } from "../../src/gpu/graph";
import { compare } from "../../src/model/compare";
import { channelFirst, channelLast, decode, loadVae, toRgba } from "../../src/model/vae";
import { loadFloats } from "../../src/model/weights";
import { readHeader } from "../../src/weights/safetensors";

const $ = <T extends HTMLElement>(selector: string) => document.querySelector<T>(selector)!;
const file = (id: string) => $<HTMLInputElement>(id).files?.[0];
const seconds = (start: number) => +((performance.now() - start) / 1000).toFixed(2);
const PASS = 0.999;

function draw(canvas: HTMLCanvasElement, rgb: Float32Array, width: number, height: number) {
  [canvas.width, canvas.height] = [width, height];
  canvas.getContext("2d")!.putImageData(new ImageData(toRgba(rgb, width * height), width, height), 0, 0);
}

$("#build").textContent = `Build ${__BUILD__}`;

$("#run").addEventListener("click", async () => {
  const status = $("#status");
  const [vaeFile, reference] = [file("#vae"), file("#reference")];
  if (!vaeFile || !reference) {
    status.textContent = "Pick the VAE file and the reference file first.";
    return;
  }
  const run = $<HTMLButtonElement>("#run");
  run.disabled = true;
  const graph = new Graph(await getDevice());
  try {
    const ref = await readHeader(reference);
    const side = Number(ref.metadata.latent);
    const latent = await loadFloats(reference, ref, "latent"); // [16, side, side]

    status.textContent = "Loading the decoder…";
    let start = performance.now();
    const weights = await loadVae(graph, vaeFile, await readHeader(vaeFile));
    const loadTime = seconds(start);

    start = performance.now();
    const z = graph.buffer(channelLast(latent, 16, side * side));
    const { image, height, width, steps } = await decode(graph, weights, z, side, side, true, (s) => (status.textContent = `Decoding: ${s}…`));
    const rgb = new Float32Array(await graph.read(image, height * width * 3 * 4));
    const runTime = seconds(start);

    const results = [];
    for (const [name, buffer] of Object.entries(steps)) {
      if (!ref.tensors.has(`step.${name}`)) continue; // ComfyUI's hooks don't see every stage
      const expected = await loadFloats(reference, ref, `step.${name}`);
      results.push({ step: name, ...compare(new Float32Array(await graph.read(buffer, expected.byteLength)), expected) });
    }
    const expectedImage = await loadFloats(reference, ref, "image"); // [3, H, W]
    results.push({ step: "image", ...compare(channelFirst(rgb, 3, height * width), expectedImage) });
    const pass = results.every((r) => r.cosine >= PASS);

    draw($("#ours"), rgb, width, height);
    draw($("#theirs"), channelLast(expectedImage, 3, height * width), width, height);
    $("#pictures").hidden = false;
    status.textContent = pass ? "✅ Every stage matches ComfyUI." : "❌ Some stages don't match ComfyUI.";
    $("#results tbody").replaceChildren(
      ...results.map((r) => {
        const row = document.createElement("tr");
        for (const text of [r.step, r.cosine.toFixed(6), r.relativeError.toExponential(2), r.cosine >= PASS ? "✅" : "❌"]) row.insertCell().textContent = text;
        return row;
      }),
    );
    $("#results").hidden = false;
    $("#report pre").textContent = JSON.stringify({ build: __BUILD__, latent: side, pass, loadTime, runTime, results }, null, 2);
    $("#report").hidden = false;
  } catch (e) {
    status.textContent = `❌ ${e instanceof Error ? e.message : String(e)}`;
  } finally {
    graph.destroy();
    run.disabled = false;
  }
});

$("#copy").addEventListener("click", () => navigator.clipboard.writeText($("#report pre").textContent ?? ""));
