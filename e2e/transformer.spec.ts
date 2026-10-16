import { expect, test } from "@playwright/test";

test("asks for both files before running the transformer", async ({ page }) => {
  await page.goto("/pages/transformer/");
  await page.getByRole("button", { name: "Run transformer", exact: true }).click();
  await expect(page.getByRole("status")).toHaveText("Pick the model file and the reference file first.");
});
