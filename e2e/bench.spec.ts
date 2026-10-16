import { expect, test } from "@playwright/test";

test("times the matmuls and estimates one image", async ({ page }) => {
  await page.goto("/pages/bench/?quick=1");
  await page.getByRole("button", { name: "Run benchmark", exact: true }).click();
  await expect(page.getByRole("status")).toHaveText(/^Done: .* TFLOPS → about \d+ s per 1024×1024 image/, { timeout: 60_000 });
  await expect(page.locator("#results tbody tr")).toHaveCount(4);
  await expect(page.locator("#report pre")).toContainText('"imageSeconds"');
  await expect(page.locator("#report pre")).toContainText('"build"');
  await expect(page.getByText(/^Build \w+$/)).toBeVisible();
});
