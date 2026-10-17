import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import * as engine from "../src";

const SRC = resolve(import.meta.dirname, "../src");

/** Every file the public entry pulls in, following relative imports. */
function reachable(file: string, seen = new Set<string>()): Set<string> {
  if (seen.has(file)) return seen;
  seen.add(file);
  const code = readFileSync(file, "utf8");
  for (const [, spec] of code.matchAll(/^\s*(?:import|export)[^"']*from\s+["']([^"']+)["']/gm)) {
    if (spec.startsWith(".")) reachable(resolve(dirname(file), `${spec}.ts`), seen);
  }
  return seen;
}

describe("public API (src/index.ts)", () => {
  it("exports what an app needs to make an image", () => {
    for (const name of ["generate", "abortable", "findModelFiles", "KREA2_FILES", "resolution", "IMAGE_SIZES", "Qwen2Tokenizer", "checkWebGPU", "torchRandn", "downloadModels", "downloadsFor"]) {
      expect(engine, name).toHaveProperty(name);
    }
  });

  it("is plain TypeScript any bundler can compile: relative imports only, no Vite-only features", () => {
    const files = reachable(resolve(SRC, "index.ts"));
    expect(files.size).toBeGreaterThan(10);
    for (const file of files) {
      const code = readFileSync(file, "utf8");
      const imports = [...code.matchAll(/^\s*(?:import|export)[^"']*from\s+["']([^"']+)["']/gm)].map((m) => m[1]);
      expect(imports.filter((s) => !s.startsWith(".")), file).toEqual([]);
      expect(code, file).not.toMatch(/import\.meta\.(env|glob)|__BUILD__|\?(raw|url|worker)["']/);
    }
  });
});
