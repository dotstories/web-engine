import { expect, test } from "@playwright/test";

test("shows the Weeping Gate recipe, its prompt and the ComfyUI reference", async ({ page }) => {
  await page.goto("/pages/dot/");
  await expect(page.locator("#recipe")).toHaveText("Krea 2 Turbo · 1928 × 1088 · seed 54460995 · 8 steps · no LoRA");
  await expect(page.locator("#prompt")).toContainText("giant pale stone hand");
  await expect(page.locator("#reference")).toHaveJSProperty("naturalWidth", 1928);
});

test("needs the model folder before making the image", async ({ page }) => {
  await page.goto("/pages/dot/");
  await page.getByRole("button", { name: "Make The Weeping Gate" }).click();
  await expect(page.locator("#status")).toHaveText("Choose your ComfyUI models folder first.");
  await expect(page.getByRole("button", { name: "Cancel" })).toBeHidden();
});

test("?models=browser finds the models in the site's private browser storage", async ({ page }) => {
  await page.goto("/pages/dot/");
  await page.evaluate(async () => {
    const models = await (await navigator.storage.getDirectory()).getDirectoryHandle("models", { create: true });
    for (const path of ["diffusion_models/krea2_turbo_fp8_scaled.safetensors", "text_encoders/qwen3vl_4b_fp8_scaled.safetensors", "vae/qwen_image_vae.safetensors"]) {
      const [folder, name] = path.split("/");
      await (await models.getDirectoryHandle(folder, { create: true })).getFileHandle(name, { create: true });
    }
  });
  await page.goto("/pages/dot/?models=browser");
  await page.getByRole("button", { name: "Choose your ComfyUI models folder" }).click();
  await expect(page.locator("#files")).toHaveText("✅ Found all 3 files.");
});
