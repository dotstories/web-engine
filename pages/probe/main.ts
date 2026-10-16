import { checkWebGPU } from "../../src/gpu/support";
import { evaluate } from "../../src/probe/gates";
import { FULL, QUICK, runProbe } from "../../src/probe/measure";

const $ = <T extends HTMLElement>(selector: string) => document.querySelector<T>(selector)!;
const sizes = new URLSearchParams(location.search).has("quick") ? QUICK : FULL;

$("#run").addEventListener("click", async () => {
  const run = $<HTMLButtonElement>("#run");
  run.disabled = true;
  const status = $("#status");
  const support = await checkWebGPU();
  const report = await runProbe(support.ok ? support.adapter : null, sizes, (s) => (status.textContent = `Measuring: ${s}…`));
  const gates = evaluate(report);

  $("#gates tbody").replaceChildren(
    ...gates.map((g) => {
      const row = document.createElement("tr");
      for (const text of [g.label, g.need, g.value, g.pass ? "✅ pass" : "❌ fail"]) {
        row.insertCell().textContent = text;
      }
      return row;
    }),
  );
  $("#gates").hidden = false;
  $("#report pre").textContent = JSON.stringify({ report, gates, userAgent: navigator.userAgent }, null, 2);
  $("#report").hidden = false;
  status.textContent = gates.every((g) => g.pass) ? "Done: all checks pass." : "Done: some checks fail.";
  run.disabled = false;
});

$("#copy").addEventListener("click", () => navigator.clipboard.writeText($("#report pre").textContent ?? ""));
