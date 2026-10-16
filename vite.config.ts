/// <reference types="vitest/config" />
import { resolve } from "node:path";
import { defineConfig } from "vite";

// Each test page is its own entry. base "./" so the build works from any path.
export default defineConfig({
  base: "./",
  build: {
    rollupOptions: {
      input: {
        index: resolve(import.meta.dirname, "index.html"),
        probe: resolve(import.meta.dirname, "pages/probe/index.html"),
        model: resolve(import.meta.dirname, "pages/model/index.html"),
      },
    },
  },
  test: {
    include: ["tests/**/*.test.ts"],
  },
});
