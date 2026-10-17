import { describe, expect, it } from "vitest";
import { loadMergedFloats } from "../src/model/weights";
import { matchLora, mergeLora } from "../src/weights/lora";
import { readHeader } from "../src/weights/safetensors";
import { makeSafetensors } from "./helpers";

const f32 = (values: number[]) => new Uint8Array(new Float32Array(values).buffer);

describe("mergeLora", () => {
  it("adds scale · up · down to the weight", () => {
    // W [1, 3], down [2, 3], up [1, 2]: up·down = [1·1 + 2·4, 1·2 + 2·5, 1·3 + 2·6] = [9, 12, 15]
    const w = mergeLora(new Float32Array([1, 1, 1]), 1, 3, new Float32Array([1, 2, 3, 4, 5, 6]), new Float32Array([1, 2]), 2, 0.5);
    expect(Array.from(w)).toEqual([5.5, 7, 8.5]);
  });
});

describe("loadMergedFloats", () => {
  // txtfusion.projector is [1, 12] in Krea 2 (the 12 → 1 text-layer mix); a small one here.
  const model = makeSafetensors({ "txtfusion.projector.weight": { dtype: "F32", shape: [1, 3], data: f32([1, 1, 1]) } });
  const lora = makeSafetensors({
    "diffusion_model.txtfusion.projector.lora_down.weight": { dtype: "F32", shape: [2, 3], data: f32([1, 2, 3, 4, 5, 6]) },
    "diffusion_model.txtfusion.projector.lora_up.weight": { dtype: "F32", shape: [1, 2], data: f32([1, 2]) },
    "diffusion_model.txtfusion.projector.alpha": { dtype: "F32", shape: [], data: f32([1]) },
  });

  it("merges the LoRA with strength · alpha / rank", async () => {
    const st = await readHeader(model);
    const matched = await matchLora(lora, await readHeader(lora), st);
    const w = await loadMergedFloats(model, st, "txtfusion.projector", { file: lora, lora: matched, strength: 0.8 });
    const s = 0.8 * (1 / 2);
    expect(Array.from(w)).toEqual(Array.from(new Float32Array([1 + s * 9, 1 + s * 12, 1 + s * 15])));
  });

  it("returns the plain weight without a LoRA or at strength 0", async () => {
    const st = await readHeader(model);
    expect(Array.from(await loadMergedFloats(model, st, "txtfusion.projector"))).toEqual([1, 1, 1]);
    const matched = await matchLora(lora, await readHeader(lora), st);
    expect(Array.from(await loadMergedFloats(model, st, "txtfusion.projector", { file: lora, lora: matched, strength: 0 }))).toEqual([1, 1, 1]);
  });
});
