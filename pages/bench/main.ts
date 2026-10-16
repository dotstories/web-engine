import { BLOCK_MATMULS, estimateImage, tokensFor } from "../../src/bench/estimate";
import { timeMatmul } from "../../src/bench/matmul";
import { getDevice } from "../../src/gpu/compute";

const $ = <T extends HTMLElement>(selector: string) => document.querySelector<T>(selector)!;
const quick = new URLSearchParams(location.search).has("quick"); // CI: tiny shapes
const tokens = quick ? 64 : tokensFor(1024, 1024);
const shapes = BLOCK_MATMULS.map((s) => (quick ? { ...s, n: s.n / 64, k: s.k / 64 } : s));

$("#run").addEventListener("click", async () => {
  const run = $<HTMLButtonElement>("#run");
  const status = $("#status");
  run.disabled = true;
  try {
    const device = await getDevice();
    const timings = [];
    for (const s of shapes) {
      status.textContent = `Timing: ${s.name}…`;
      timings.push({ ...s, ms: await timeMatmul(device, { m: tokens, n: s.n, k: s.k }) });
    }
    const estimate = estimateImage(timings, tokens);

    $("#results tbody").replaceChildren(
      ...timings.map((t) => {
        const row = document.createElement("tr");
        const tflops = (2 * tokens * t.n * t.k) / (t.ms * 1e9);
        for (const text of [`${t.name} ×${t.count}`, `${tokens}×${t.k} · ${t.k}×${t.n}`, t.ms.toFixed(1), tflops.toFixed(1)]) {
          row.insertCell().textContent = text;
        }
        return row;
      }),
    );
    $("#results").hidden = false;
    status.textContent = `Done: ${estimate.tflops.toFixed(1)} TFLOPS → about ${Math.round(estimate.imageSeconds)} s per 1024×1024 image (8 steps).`;
    $("#report pre").textContent = JSON.stringify({ tokens, timings, estimate, userAgent: navigator.userAgent }, null, 2);
    $("#report").hidden = false;
  } catch (e) {
    status.textContent = `❌ ${e instanceof Error ? e.message : String(e)}`;
  } finally {
    run.disabled = false;
  }
});

$("#copy").addEventListener("click", () => navigator.clipboard.writeText($("#report pre").textContent ?? ""));
