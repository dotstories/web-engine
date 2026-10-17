import { expect, test } from "@playwright/test";

test("needs the model folder before generating", async ({ page }) => {
  await page.goto("/pages/generate/");
  await expect(page.getByLabel("Prompt")).toHaveValue(/dark fantasy castle/);
  await page.getByRole("button", { name: "Generate", exact: true }).click();
  await expect(page.locator("#status")).toHaveText("Choose your ComfyUI models folder first.");
});

test("offers the LoRA with the Dark Fantasy Kit strength", async ({ page }) => {
  await page.goto("/pages/generate/");
  await expect(page.getByLabel("LoRA")).toHaveValue("");
  await expect(page.getByLabel("Strength")).toHaveValue("0.8");
});

test("offers the Dark Fantasy Kit size (16:9, 2 MP)", async ({ page }) => {
  await page.goto("/pages/generate/");
  const size = page.getByLabel("Size");
  await expect(size).toHaveValue("1024x1024");
  await expect(size.locator("option")).toContainText(["512 × 512", "1024 × 1024", "1368 × 768", "1928 × 1088 (16:9, 2 MP, Dark Fantasy Kit)"]);
});
