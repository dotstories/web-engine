import { expect, test } from "@playwright/test";

test("asks for both files before running", async ({ page }) => {
  await page.goto("/pages/block/");
  await expect(page.getByLabel("Model file")).toBeVisible();
  await expect(page.getByLabel("Reference file")).toBeVisible();
  await page.getByRole("button", { name: "Run block", exact: true }).click();
  await expect(page.getByRole("status")).toHaveText("Pick the model file and the reference file first.");
});
