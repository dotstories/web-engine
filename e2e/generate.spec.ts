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
