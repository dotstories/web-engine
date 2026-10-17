import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { WEEPING_GATE } from "../pages/dot/recipe";
import { resolution } from "../src/pipeline/sizes";
import { kreaPrompt } from "../src/text/prompt";
import { Qwen2Tokenizer } from "../src/text/tokenizer";

const file = (name: string) => readFileSync(new URL(`../public/${name}`, import.meta.url));

describe("The Weeping Gate recipe", () => {
  it("keeps the prompt byte for byte (ComfyUI's rewritten text, no trailing newline)", () => {
    expect(createHash("sha256").update(file("dots/weeping-gate/prompt.txt")).digest("hex")).toBe(WEEPING_GATE.promptSha256);
  });

  it("uses the Kit's size: 16:9 at 2 MP, multiple of 8", () => {
    expect(resolution([16, 9], 2, 8)).toEqual({ width: WEEPING_GATE.width, height: WEEPING_GATE.height });
  });

  const tokenizer = new Qwen2Tokenizer(JSON.parse(file("tokenizers/qwen2/vocab.json").toString()), file("tokenizers/qwen2/merges.txt").toString());
  const text = file("dots/weeping-gate/prompt.txt").toString();

  it("is exactly the 512 tokens the Kit's prompt rewrite stopped at (max_length 512): same tokenizer as ComfyUI", () => {
    expect(tokenizer.encode(text)).toHaveLength(512);
  });

  it("encodes the whole prompt in Krea 2's template, no cut-off (ComfyUI's Krea 2 tokenizer has no limit)", () => {
    expect(tokenizer.encode(kreaPrompt(text))).toHaveLength(551);
  });
});
