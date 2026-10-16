import { expect, test } from "@playwright/test";

test("asks for both files before encoding", async ({ page }) => {
  await page.goto("/pages/text/");
  await page.getByRole("button", { name: "Encode", exact: true }).click();
  await expect(page.getByRole("status")).toHaveText("Pick the text encoder file and the reference file first.");
});

test("serves the tokenizer files next to the pages", async ({ request }) => {
  expect((await request.get("/tokenizers/qwen2/merges.txt")).ok()).toBe(true);
  expect((await request.get("/tokenizers/qwen2/vocab.json")).ok()).toBe(true);
});
