import { expect, test } from "@playwright/test";

test("probes the computer and shows every gate", async ({ page }) => {
  await page.goto("/pages/probe/?quick=1");
  await page.getByRole("button", { name: "Run probe", exact: true }).click();
  await expect(page.getByRole("status")).toHaveText(/^Done/, { timeout: 60_000 });

  const rows = page.locator("#gates tbody tr");
  await expect(rows).toHaveCount(6);
  await expect(rows.filter({ hasText: "WebGPU GPU" })).toContainText("✅ pass");
  await expect(rows.filter({ hasText: "Storage read speed" })).not.toContainText("not measured");
  await expect(page.locator("#report pre")).toContainText('"maxStorageBufferBindingSize"');
});
