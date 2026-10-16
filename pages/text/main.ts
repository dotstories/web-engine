import { getDevice } from "../../src/gpu/compute";
import { Graph } from "../../src/gpu/graph";
import { compare } from "../../src/model/compare";
import { QWEN3VL_4B, TAPS, encode, loadTextEncoder } from "../../src/model/text-encoder";
import { loadFloats } from "../../src/model/weights";
import { kreaPrompt } from "../../src/text/prompt";
import { Qwen2Tokenizer } from "../../src/text/tokenizer";
import { readHeader } from "../../src/weights/safetensors";

const $ = <T extends HTMLElement>(selector: string) => document.querySelector<T>(selector)!;
const file = (id: string) => $<HTMLInputElement>(id).files?.[0];
const seconds = (start: number) => +((performance.now() - start) / 1000).toFixed(2);
const PASS = 0.999;
const tokenizerFile = (name: string) => new URL(`../../tokenizers/qwen2/${name}`, location.href);

$("#build").textContent = `Build ${__BUILD__}`;

$("#run").addEventListener("click", async () => {
  const status = $("#status");
  const [te, reference] = [file("#te"), file("#reference")];
  if (!te || !reference) {
    status.textContent = "Pick the text encoder file and the reference file first.";
    return;
  }
  const run = $<HTMLButtonElement>("#run");
  run.disabled = true;
  const graph = new Graph(await getDevice());
  try {
    const ref = await readHeader(reference);
    const prompt = ref.metadata.prompt;
    const expectedIds = Array.from(await loadFloats(reference, ref, "ids"));
    const expected = await loadFloats(reference, ref, "cond");

    status.textContent = "Tokenizing…";
    const tokenizer = await Qwen2Tokenizer.load(tokenizerFile("vocab.json"), tokenizerFile("merges.txt"));
    const ids = tokenizer.encode(kreaPrompt(prompt));
    const sameTokens = JSON.stringify(ids) === JSON.stringify(expectedIds);

    let start = performance.now();
    const weights = await loadTextEncoder(graph, te, await readHeader(te), (done, total) => (status.textContent = `Loading layer ${done} of ${total}…`));
    const loadTime = seconds(start);

    start = performance.now();
    const { cond, kept } = await encode(graph, weights, ids, (i) => (status.textContent = `Encoding: layer ${i + 1}…`));
    const runTime = seconds(start);

    const hidden = QWEN3VL_4B.hidden;
    const layer = (x: Float32Array, l: number) =>
      Float32Array.from({ length: kept * hidden }, (_, i) => x[(Math.floor(i / hidden) * TAPS.length + l) * hidden + (i % hidden)]);
    const results = TAPS.map((tap, l) => ({ step: `layer ${tap}`, ...compare(layer(cond, l), layer(expected, l)) }));
    results.push({ step: "all", ...compare(cond, expected) });
    const pass = sameTokens && results.every((r) => r.cosine >= PASS);

    status.textContent = pass ? "✅ Tokens and every tapped layer match ComfyUI." : sameTokens ? "❌ Some layers don't match ComfyUI." : "❌ The tokens differ from ComfyUI.";
    $("#results tbody").replaceChildren(
      ...results.map((r) => {
        const row = document.createElement("tr");
        for (const text of [r.step, r.cosine.toFixed(6), r.relativeError.toExponential(2), r.cosine >= PASS ? "✅" : "❌"]) row.insertCell().textContent = text;
        return row;
      }),
    );
    $("#results").hidden = false;
    $("#report pre").textContent = JSON.stringify({ build: __BUILD__, prompt, tokens: ids.length, kept, sameTokens, pass, loadTime, runTime, results }, null, 2);
    $("#report").hidden = false;
  } catch (e) {
    status.textContent = `❌ ${e instanceof Error ? e.message : String(e)}`;
  } finally {
    graph.destroy();
    run.disabled = false;
  }
});

$("#copy").addEventListener("click", () => navigator.clipboard.writeText($("#report pre").textContent ?? ""));
