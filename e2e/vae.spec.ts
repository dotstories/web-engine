import { expect, test } from "@playwright/test";

test("asks for both files before decoding", async ({ page }) => {
  await page.goto("/pages/vae/");
  await page.getByRole("button", { name: "Decode", exact: true }).click();
  await expect(page.getByRole("status")).toHaveText("Pick the VAE file and the reference file first.");
});
