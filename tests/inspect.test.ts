import { describe, expect, it } from "vitest";
import { blockGroups, fp8Weights, summarize } from "../src/weights/inspect";
import { readHeader } from "../src/weights/safetensors";
import { makeSafetensors } from "./helpers";

const t = (dtype: string, shape: number[]) => ({ dtype, shape, data: new Uint8Array(shape.reduce((a, b) => a * b, 1) * (dtype === "F32" ? 4 : dtype === "BF16" ? 2 : 1)) });

const krea = () =>
  readHeader(
    makeSafetensors({
      "model.diffusion_model.txtfusion.projector.weight": t("BF16", [4, 2]),
      "model.diffusion_model.blocks.0.attn.qkv.weight": t("F8_E4M3", [6, 2]),
      "model.diffusion_model.blocks.0.attn.qkv.scale_weight": t("F32", [1]),
      "model.diffusion_model.blocks.1.attn.qkv.weight": t("F8_E4M3", [6, 2]),
      "model.diffusion_model.blocks.1.attn.qkv.weight_scale": t("F32", [1]),
      "model.diffusion_model.blocks.1.norm.weight": t("BF16", [2]),
      "model.diffusion_model.final.weight": t("F8_E4M3", [2, 2]),
      scaled_fp8: t("F8_E4M3", [0]),
    }),
  );

describe("fp8Weights", () => {
  it("pairs each fp8 weight with its scale (ComfyUI fp8_scaled), either name", async () => {
    const pairs = fp8Weights(await krea());
    const short = (name?: string) => name?.replace("model.diffusion_model.", "") ?? null;
    expect(pairs.map((p) => [short(p.weight.name), short(p.scale?.name)])).toEqual([
      ["blocks.0.attn.qkv.weight", "blocks.0.attn.qkv.scale_weight"],
      ["blocks.1.attn.qkv.weight", "blocks.1.attn.qkv.weight_scale"],
      ["final.weight", null],
    ]);
  });

  it("skips the scaled_fp8 marker", async () => {
    expect(fp8Weights(await krea()).some((p) => p.weight.name === "scaled_fp8")).toBe(false);
  });
});

describe("blockGroups", () => {
  it("finds repeated blocks and how many there are", () => {
    const names = ["m.blocks.0.a", "m.blocks.0.b", "m.blocks.1.a", "m.blocks.2.a", "m.txt.0.x", "other"];
    expect(blockGroups(names)).toEqual([
      { prefix: "m.blocks", count: 3 },
      { prefix: "m.txt", count: 1 },
    ]);
  });

  it("uses the outermost number for nested blocks", () => {
    expect(blockGroups(["m.blocks.0.mlp.1.w", "m.blocks.1.mlp.0.w"])).toEqual([{ prefix: "m.blocks", count: 2 }]);
  });
});

describe("summarize", () => {
  it("recognizes Krea 2 and counts what is inside", async () => {
    const s = summarize(await krea());
    expect(s).toMatchObject({
      tensors: 8,
      isKrea: true,
      dtypes: { BF16: 2, F8_E4M3: 4, F32: 2 },
      fp8: { weights: 3, scaled: 2 },
      blocks: [{ prefix: "model.diffusion_model.blocks", count: 2 }],
    });
    expect(s.bytes).toBe(8 * 2 + 12 + 4 + 12 + 4 + 2 * 2 + 4);
  });

  it("lists the first block's tensors so the layout can be read", async () => {
    const s = summarize(await krea());
    expect(s.firstBlock).toEqual([
      { name: "model.diffusion_model.blocks.0.attn.qkv.scale_weight", dtype: "F32", shape: [1] },
      { name: "model.diffusion_model.blocks.0.attn.qkv.weight", dtype: "F8_E4M3", shape: [6, 2] },
    ]);
  });

  it("says when a file is not Krea 2", async () => {
    const s = summarize(await readHeader(makeSafetensors({ "unet.weight": t("F32", [1]) })));
    expect(s.isKrea).toBe(false);
  });
});
