/// <reference types="vitest/config" />
/// <reference types="@vitest/browser/providers/playwright" />
import { resolve } from "node:path";
import { defineConfig } from "vite";
import { webgpuFlags } from "./chrome-webgpu";

// Each test page is its own entry. base "./" so the build works from any path.
export default defineConfig({
  base: "./",
  build: {
    rollupOptions: {
      input: {
        index: resolve(import.meta.dirname, "index.html"),
        probe: resolve(import.meta.dirname, "pages/probe/index.html"),
        model: resolve(import.meta.dirname, "pages/model/index.html"),
        bench: resolve(import.meta.dirname, "pages/bench/index.html"),
      },
    },
  },
  test: {
    projects: [
      { extends: true, test: { name: "unit", include: ["tests/**/*.test.ts"], exclude: ["tests/gpu/**"] } },
      {
        // GPU kernels run in real Chrome with WebGPU.
        extends: true,
        test: {
          name: "gpu",
          include: ["tests/gpu/**/*.test.ts"],
          browser: {
            enabled: true,
            provider: "playwright",
            headless: true,
            instances: [{ browser: "chromium", launch: { args: webgpuFlags } }],
          },
        },
      },
    ],
  },
});
