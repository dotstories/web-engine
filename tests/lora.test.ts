import { describe, expect, it } from "vitest";
import { describeLora, diffusersName, loraMatrix, matchLora } from "../src/weights/lora";
import { readHeader } from "../src/weights/safetensors";
import { makeSafetensors } from "./helpers";

const SIZES: Record<string, number> = { F32: 4, BF16: 2, F16: 2, F8_E4M3: 1 };
const t = (dtype: string, shape: number[], data?: Uint8Array<ArrayBuffer>) => ({
  dtype,
  shape,
  data: data ?? new Uint8Array(shape.reduce((a, b) => a * b, 1) * SIZES[dtype]),
});
const f32 = (value: number) => t("F32", [], new Uint8Array(new Float32Array([value]).buffer));

/** A tiny Krea-like model: wq [8, 4] fp8, mlp.down [4, 6] bf16, a norm. */
const model = () =>
  readHeader(
    makeSafetensors({
      "blocks.0.attn.wq.weight": t("F8_E4M3", [8, 4]),
      "blocks.0.attn.wq.weight_scale": t("F32", [1]),
      "blocks.0.attn.wo.weight": t("F8_E4M3", [4, 8]),
      "blocks.0.mlp.down.weight": t("BF16", [4, 6]),
      "blocks.0.prenorm.scale": t("F32", [4]),
      "txtmlp.1.weight": t("BF16", [4, 4]),
    }),
  );

async function match(tensors: Parameters<typeof makeSafetensors>[0]) {
  const file = makeSafetensors(tensors);
  return matchLora(file, await readHeader(file), await model());
}

describe("diffusersName", () => {
  it("maps ComfyUI Krea 2 names to diffusers names (comfy/utils.py krea2_to_diffusers)", () => {
    expect(diffusersName("blocks.3.attn.wq")).toEqual(["transformer_blocks.3.attn.to_q"]);
    expect(diffusersName("blocks.3.attn.wo")).toEqual(["transformer_blocks.3.attn.to_out.0", "transformer_blocks.3.attn.to_out"]);
    expect(diffusersName("blocks.0.mlp.down")).toEqual(["transformer_blocks.0.ff.down"]);
    expect(diffusersName("txtfusion.refiner_blocks.1.attn.gate")).toEqual(["text_fusion.refiner_blocks.1.attn.to_gate"]);
    expect(diffusersName("txtmlp.1")).toEqual(["txt_in.linear_1"]);
    expect(diffusersName("last.linear")).toEqual(["final_layer.linear"]);
    expect(diffusersName("blocks.0.prenorm")).toEqual([]);
  });
});

describe("matchLora", () => {
  it("reads kohya names (lora_unet_, lora_up/lora_down, alpha) and scales by alpha / rank", async () => {
    const lora = await match({
      "lora_unet_blocks_0_attn_wq.lora_down.weight": t("BF16", [2, 4]),
      "lora_unet_blocks_0_attn_wq.lora_up.weight": t("BF16", [8, 2]),
      "lora_unet_blocks_0_attn_wq.alpha": f32(1),
    });
    const wq = lora.targets.get("blocks.0.attn.wq");
    expect(wq?.down.name).toBe("lora_unet_blocks_0_attn_wq.lora_down.weight");
    expect(wq?.up.name).toBe("lora_unet_blocks_0_attn_wq.lora_up.weight");
    expect(wq).toMatchObject({ rank: 2, scale: 0.5 });
    expect(lora.unused).toEqual([]);
  });

  it("reads diffusers / peft names (transformer.*, lora_A/lora_B) without alpha as scale 1", async () => {
    const lora = await match({
      "transformer.transformer_blocks.0.attn.to_out.0.lora_A.weight": t("BF16", [2, 8]),
      "transformer.transformer_blocks.0.attn.to_out.0.lora_B.weight": t("BF16", [4, 2]),
      "transformer.txt_in.linear_1.lora_A.weight": t("F32", [3, 4]),
      "transformer.txt_in.linear_1.lora_B.weight": t("F32", [4, 3]),
    });
    expect([...lora.targets.keys()].sort()).toEqual(["blocks.0.attn.wo", "txtmlp.1"]);
    expect(lora.targets.get("blocks.0.attn.wo")).toMatchObject({ rank: 2, scale: 1 });
  });

  it("reads ComfyUI names (diffusion_model.*) and the other up/down spellings ComfyUI accepts", async () => {
    const lora = await match({
      "diffusion_model.blocks.0.attn.wq.lora.down.weight": t("BF16", [2, 4]),
      "diffusion_model.blocks.0.attn.wq.lora.up.weight": t("BF16", [8, 2]),
      "blocks.0.mlp.down.lora_A": t("BF16", [2, 6]),
      "blocks.0.mlp.down.lora_B": t("BF16", [4, 2]),
      "blocks.0.mlp.down.alpha": f32(8),
    });
    expect([...lora.targets.keys()].sort()).toEqual(["blocks.0.attn.wq", "blocks.0.mlp.down"]);
    expect(lora.targets.get("blocks.0.mlp.down")?.scale).toBe(4);
  });

  it("lists keys it did not use, like ComfyUI's 'lora key not loaded'", async () => {
    const lora = await match({
      "diffusion_model.blocks.0.attn.wq.lora_A.weight": t("BF16", [2, 4]),
      "diffusion_model.blocks.0.attn.wq.lora_B.weight": t("BF16", [8, 2]),
      "diffusion_model.blocks.9.attn.wq.lora_A.weight": t("BF16", [2, 4]),
      "diffusion_model.blocks.9.attn.wq.lora_B.weight": t("BF16", [8, 2]),
      "diffusion_model.blocks.0.prenorm.diff": t("F32", [4]),
    });
    expect([...lora.targets.keys()]).toEqual(["blocks.0.attn.wq"]);
    expect(lora.unused).toEqual([
      "diffusion_model.blocks.0.prenorm.diff",
      "diffusion_model.blocks.9.attn.wq.lora_A.weight",
      "diffusion_model.blocks.9.attn.wq.lora_B.weight",
    ]);
  });

  it("rejects up/down shapes that do not fit the model weight", async () => {
    await expect(
      match({
        "diffusion_model.blocks.0.attn.wq.lora_A.weight": t("BF16", [2, 5]),
        "diffusion_model.blocks.0.attn.wq.lora_B.weight": t("BF16", [8, 2]),
      }),
    ).rejects.toThrow("blocks.0.attn.wq: LoRA down [2,5] and up [8,2] do not fit the weight [8,4]");
  });

  it("rejects LoRA kinds we do not run yet (DoRA, LoCon mid)", async () => {
    await expect(
      match({
        "diffusion_model.blocks.0.attn.wq.lora_A.weight": t("BF16", [2, 4]),
        "diffusion_model.blocks.0.attn.wq.lora_B.weight": t("BF16", [8, 2]),
        "diffusion_model.blocks.0.attn.wq.dora_scale": t("BF16", [8, 1]),
      }),
    ).rejects.toThrow("blocks.0.attn.wq: DoRA / mid LoRA weights are not supported yet");
  });

  it("rejects a file with no LoRA weights for this model", async () => {
    await expect(match({ "something.else": t("F32", [1]) })).rejects.toThrow("No LoRA weights for this model in the file");
  });
});

describe("loraMatrix", () => {
  const f32bytes = (values: number[]) => new Uint8Array(new Float32Array(values).buffer);

  it("keeps bf16 as bf16 when the rank is a multiple of 4", () => {
    const bytes = new Uint8Array(4 * 2 * 2).map((_, i) => i);
    const m = loraMatrix("BF16", bytes, 4, 2, "rows");
    expect(m).toMatchObject({ dtype: "bf16", rows: 4, cols: 2 });
    expect(new Uint8Array(m.data.buffer)).toEqual(bytes);
  });

  it("pads down [rank, k] with zero rows and up [n, rank] with zero columns, as f32", () => {
    const down = loraMatrix("F32", f32bytes([1, 2, 3, 4, 5, 6]), 3, 2, "rows"); // rank 3, k 2
    expect(down).toMatchObject({ dtype: "f32", rows: 4, cols: 2 });
    expect(Array.from(down.data)).toEqual([1, 2, 3, 4, 5, 6, 0, 0]);

    const up = loraMatrix("F32", f32bytes([1, 2, 3, 4, 5, 6]), 2, 3, "cols"); // n 2, rank 3
    expect(up).toMatchObject({ dtype: "f32", rows: 2, cols: 4 });
    expect(Array.from(up.data)).toEqual([1, 2, 3, 0, 4, 5, 6, 0]);
  });

  it("turns F16 into f32", () => {
    const half = new Uint8Array(new Uint16Array([0x3c00, 0xc000, 0x3800, 0]).buffer); // 1, -2, 0.5, 0
    const m = loraMatrix("F16", half, 4, 1, "rows");
    expect(m.dtype).toBe("f32");
    expect(Array.from(m.data)).toEqual([1, -2, 0.5, 0]);
  });
});

describe("describeLora", () => {
  it("sums up layers, dtypes, ranks, bytes and unused keys", async () => {
    const lora = await match({
      "diffusion_model.blocks.0.attn.wq.lora_A.weight": t("BF16", [2, 4]),
      "diffusion_model.blocks.0.attn.wq.lora_B.weight": t("BF16", [8, 2]),
      "diffusion_model.blocks.0.mlp.down.lora_A.weight": t("F32", [4, 6]),
      "diffusion_model.blocks.0.mlp.down.lora_B.weight": t("F32", [4, 4]),
      "diffusion_model.something.else": t("F32", [1]),
    });
    expect(describeLora(lora)).toEqual({ layers: 2, dtypes: ["BF16", "F32"], ranks: [2, 4], bytes: 2 * (16 + 16) + 4 * (24 + 16), unused: 1 });
  });
});
