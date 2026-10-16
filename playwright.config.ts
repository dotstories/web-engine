import { defineConfig, devices } from "@playwright/test";
import { webgpuFlags } from "./chrome-webgpu";

export default defineConfig({
  testDir: "e2e",
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? "github" : "list",
  use: { baseURL: "http://localhost:5173", trace: "retain-on-failure" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"], launchOptions: { args: webgpuFlags } } }],
  webServer: { command: "pnpm exec vite --port 5173 --strictPort", url: "http://localhost:5173", reuseExistingServer: !process.env.CI },
});
