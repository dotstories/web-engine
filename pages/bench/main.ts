import { KREA2_ATTENTION, timeAttention } from "../../src/bench/attention";
import { BLOCK_MATMULS, estimateImage, tokensFor } from "../../src/bench/estimate";
import { timeMatmul } from "../../src/bench/matmul";
import { peakSupported, timePeak, type PeakKind } from "../../src/bench/peak";
import { getDevice } from "../../src/gpu/compute";

const $ = <T extends HTMLElement>(selector: string) => document.querySelector<T>(selector)!;
const quick = new URLSearchParams(location.search).has("quick"); // CI: tiny shapes
const tokens = quick ? 64 : tokensFor(1024, 1024);
const shapes = BLOCK_MATMULS.map((s) => (quick ? { ...s, n: s.n / 64, k: s.k / 64 } : s));

$("#build").textContent = `Build ${__BUILD__}`;

$("#run").addEventListener("click", async () => {
  const run = $<HTMLButtonElement>("#run");
  const status = $("#status");
  run.disabled = true;
  try {
    const device = await getDevice();
    const peaks: { kind: PeakKind; ms: number; tflops: number }[] = [];
    for (const kind of ["f32", "f16", "i8"] as const) {
      if (!peakSupported(device, kind)) continue;
      status.textContent = `Timing: ${kind} peak…`;
      peaks.push({ kind, ...(await timePeak(device, kind, quick ? { workgroups: 4, iters: 16, warmupMs: 20 } : {})) });
    }
    $("#peaks tbody").replaceChildren(
      ...peaks.map((p) => {
        const row = document.createElement("tr");
        for (const text of [p.kind, p.ms.toFixed(2), p.tflops.toFixed(1)]) row.insertCell().textContent = text;
        return row;
      }),
    );
    $("#peaks").hidden = false;
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
    status.textContent = "Timing: attention at 1928×1088…";
    const attention = await timeAttention(device, quick ? { tokens: 70, heads: 4, kvHeads: 2, dim: 16 } : KREA2_ATTENTION);
    $("#attention tbody").replaceChildren(
      ...attention.map((a) => {
        const row = document.createElement("tr");
        for (const text of [a.name, a.ms.toFixed(1), a.tflops?.toFixed(1) ?? ""]) row.insertCell().textContent = text;
        return row;
      }),
    );
    $("#attention").hidden = false;
    const f32Peak = peaks.find((p) => p.kind === "f32")?.tflops ?? 0;
    const share = f32Peak ? ` The matmul reaches ${Math.round((100 * estimate.tflops) / f32Peak)}% of the f32 peak.` : "";
    status.textContent = `Done: ${estimate.tflops.toFixed(1)} TFLOPS → about ${Math.round(estimate.imageSeconds)} s per 1024×1024 image (8 steps).${share}`;
    $("#report pre").textContent = JSON.stringify({ build: __BUILD__, tokens, peaks, timings, estimate, attention, userAgent: navigator.userAgent }, null, 2);
    $("#report").hidden = false;
  } catch (e) {
    status.textContent = `❌ ${e instanceof Error ? e.message : String(e)}`;
  } finally {
    run.disabled = false;
  }
});

$("#copy").addEventListener("click", () => navigator.clipboard.writeText($("#report pre").textContent ?? ""));
