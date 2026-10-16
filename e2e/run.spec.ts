import { expect, test } from "@playwright/test";

test("asks for the model file before the speed run", async ({ page }) => {
  await page.goto("/pages/run/");
  await page.getByRole("button", { name: "Run 8 steps", exact: true }).click();
  await expect(page.getByRole("status")).toHaveText("Pick the model file first.");
});
