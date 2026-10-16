import { expect, test } from "@playwright/test";

test("finds a WebGPU adapter", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("status")).toHaveText(/^WebGPU ready/);
});
