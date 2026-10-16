import { formatBytes } from "../../src/probe/gates";
import { toFloat32 } from "../../src/weights/fp8";
import { fp8Weights, summarize } from "../../src/weights/inspect";
import { readHeader, readTensor, type Safetensors } from "../../src/weights/safetensors";

const $ = <T extends HTMLElement>(selector: string) => document.querySelector<T>(selector)!;

/** The first values of the first fp8 weight, scale applied: they should be small, finite numbers. */
async function sample(file: Blob, st: Safetensors) {
  const first = fp8Weights(st)[0];
  if (!first) return null;
  const raw = Array.from(toFloat32(first.weight.dtype, await readTensor(file, first.weight, 8)));
  const scale = first.scale ? toFloat32(first.scale.dtype, await readTensor(file, first.scale, 4))[0] : 1;
  return { tensor: first.weight.name, scale, values: raw.map((v) => +(v * scale).toPrecision(4)) };
}

$("#file").addEventListener("change", async (event) => {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (!file) return;
  const status = $("#status");
  $("#summary").hidden = $("#report").hidden = true;
  try {
    const start = performance.now();
    const st = await readHeader(file);
    const summary = summarize(st);
    const values = await sample(file, st);
    const ms = Math.round(performance.now() - start);

    status.textContent = summary.isKrea ? "✅ This is Krea 2." : "❌ This is not a Krea 2 file.";
    const rows: [string, string][] = [
      ["File", `${file.name} · ${formatBytes(file.size)}`],
      ["Tensors", `${summary.tensors} · ${formatBytes(summary.bytes)}`],
      ["Types", Object.entries(summary.dtypes).map(([d, n]) => `${d} × ${n}`).join(", ")],
      ["fp8 weights", `${summary.fp8.weights} (${summary.fp8.scaled} with a scale)`],
      ["Blocks", summary.blocks.slice(0, 4).map((b) => `${b.prefix} × ${b.count}`).join(", ") || "none"],
      ["Sample", values ? values.values.join(", ") : "no fp8 weights"],
      ["Read in", `${ms} ms`],
    ];
    $("#summary tbody").replaceChildren(
      ...rows.map(([label, value]) => {
        const row = document.createElement("tr");
        row.insertCell().textContent = label;
        row.insertCell().textContent = value;
        return row;
      }),
    );
    $("#summary").hidden = false;
    $("#report pre").textContent = JSON.stringify({ file: { name: file.name, size: file.size }, summary, sample: values, ms }, null, 2);
    $("#report").hidden = false;
  } catch (e) {
    status.textContent = `❌ Can't read this file: ${e instanceof Error ? e.message : String(e)}`;
  }
});

$("#copy").addEventListener("click", () => navigator.clipboard.writeText($("#report pre").textContent ?? ""));
