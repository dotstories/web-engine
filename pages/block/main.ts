import { getDevice } from "../../src/gpu/compute";
import { Graph } from "../../src/gpu/graph";
import { ropeTables } from "../../src/kernels/rope";
import { block, KREA_BLOCK } from "../../src/model/block";
import { compare } from "../../src/model/compare";
import { loadBlock, loadFloats } from "../../src/model/weights";
import { readHeader } from "../../src/weights/safetensors";

const $ = <T extends HTMLElement>(selector: string) => document.querySelector<T>(selector)!;
const file = (id: string) => $<HTMLInputElement>(id).files?.[0];
const seconds = (start: number) => +((performance.now() - start) / 1000).toFixed(2);

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
    const index = Number(ref.metadata.block ?? 0);
    const x = await loadFloats(reference, ref, "x");
    const temb = await loadFloats(reference, ref, "temb");
    const expected = await loadFloats(reference, ref, "out");
    const flat = await loadFloats(reference, ref, "positions");
    const pos = Array.from({ length: flat.length / 3 }, (_, t) => [flat[3 * t], flat[3 * t + 1], flat[3 * t + 2]]);
    const tokens = pos.length;

    status.textContent = `Loading block ${index} onto the GPU…`;
    let start = performance.now();
    const weights = await loadBlock(graph, model, await readHeader(model), index);
    const loadTime = seconds(start);

    status.textContent = "Running the block…";
    start = performance.now();
    const tables = ropeTables(pos);
    const rope = { cos: graph.buffer(tables.cos), sin: graph.buffer(tables.sin), pairs: tables.pairs, tokens };
    const out = block(graph, weights, KREA_BLOCK, graph.buffer(x), temb, rope);
    const result = new Float32Array(await graph.read(out, tokens * KREA_BLOCK.hidden * 4));
    const runTime = seconds(start);

    const r = compare(result, expected);
    const pass = r.cosine >= 0.999;
    status.textContent = pass ? "✅ The block matches the reference." : "❌ The block does not match the reference.";
    const rows: [string, string][] = [
      ["Block", `${index} · ${tokens} tokens`],
      ["Cosine similarity", r.cosine.toFixed(6)],
      ["Relative error", r.relativeError.toExponential(2)],
      ["Largest error", r.maxError.toExponential(2)],
      ["Load / run", `${loadTime} s / ${runTime} s`],
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
    const sample = { result: Array.from(result.slice(0, 6)), expected: Array.from(expected.slice(0, 6)) };
    $("#report pre").textContent = JSON.stringify({ build: __BUILD__, block: index, tokens, ...r, pass, loadTime, runTime, sample }, null, 2);
    $("#report").hidden = false;
  } catch (e) {
    status.textContent = `❌ ${e instanceof Error ? e.message : String(e)}`;
  } finally {
    graph.destroy();
    run.disabled = false;
  }
});

$("#copy").addEventListener("click", () => navigator.clipboard.writeText($("#report pre").textContent ?? ""));
