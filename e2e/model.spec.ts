import { expect, test } from "@playwright/test";
import { makeSafetensors } from "../tests/helpers";

const f32 = (v: number) => new Uint8Array(new Float32Array([v]).buffer);

async function fakeKrea() {
  const file = makeSafetensors({
    "model.diffusion_model.txtfusion.projector.weight": { dtype: "BF16", shape: [2], data: new Uint8Array(4) },
    "model.diffusion_model.blocks.0.attn.qkv.weight": { dtype: "F8_E4M3", shape: [2, 2], data: new Uint8Array([0x38, 0x40, 0xb8, 0]) },
    "model.diffusion_model.blocks.0.attn.qkv.scale_weight": { dtype: "F32", shape: [1], data: f32(0.5) },
  });
  return Buffer.from(await file.arrayBuffer());
}

test("opens a Krea 2 file and shows what is inside", async ({ page }) => {
  await page.goto("/pages/model/");
  await page.getByLabel("Model file").setInputFiles({ name: "krea.safetensors", mimeType: "application/octet-stream", buffer: await fakeKrea() });
  await expect(page.getByRole("status")).toHaveText("✅ This is Krea 2.");
  await expect(page.getByRole("row", { name: /fp8 weights/ })).toContainText("1 (1 with a scale)");
  await expect(page.getByRole("row", { name: /Sample/ })).toContainText("0.5, 1, -0.5, 0"); // fp8 × scale
  await expect(page.locator("#report pre")).toContainText('"isKrea": true');
});

test("explains a file it can't read", async ({ page }) => {
  await page.goto("/pages/model/");
  await page.getByLabel("Model file").setInputFiles({ name: "x.safetensors", mimeType: "application/octet-stream", buffer: Buffer.from("not a model") });
  await expect(page.getByRole("status")).toContainText("❌ Can't read this file");
});
