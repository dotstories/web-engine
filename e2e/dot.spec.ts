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
