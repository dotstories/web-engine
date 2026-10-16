import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { kreaPrompt, templateEnd } from "../src/text/prompt";
import { Qwen2Tokenizer } from "../src/text/tokenizer";

const dir = new URL("../public/tokenizers/qwen2/", import.meta.url);
const tokenizer = new Qwen2Tokenizer(JSON.parse(readFileSync(new URL("vocab.json", dir), "utf8")), readFileSync(new URL("merges.txt", dir), "utf8"));

// From tools/reference/text.py: ComfyUI's Krea 2 tokenizer on the default prompt.
const PROMPT = "a dark fantasy castle on a cliff at night, lightning, ravens";
const COMFYUI = [
  151644, 8948, 198, 74785, 279, 2168, 553, 44193, 279, 1894, 11, 6083, 11, 1379, 11, 10434, 11, 12194, 11, 1467, 11, 27979, 11871, 315,
  279, 6171, 323, 4004, 25, 151645, 198, 151644, 872, 198, 64, 6319, 18378, 32584, 389, 264, 43006, 518, 3729, 11, 32438, 11, 42543,
  724, 151645, 198, 151644, 77091, 198,
];

describe("Qwen2Tokenizer", () => {
  it("gives exactly ComfyUI's tokens for a Krea 2 prompt", () => {
    expect(tokenizer.encode(kreaPrompt(PROMPT))).toEqual(COMFYUI);
  });

  it("keeps special tokens whole and handles spaces, digits, punctuation and non-ASCII text", () => {
    const ids = tokenizer.encode("<|im_start|>Hello  world 2026!\nCafé 東京 🐉<|im_end|>");
    expect(ids[0]).toBe(151644);
    expect(ids.at(-1)).toBe(151645);
    expect(ids.length).toBeGreaterThan(8);
  });
});

describe("templateEnd", () => {
  it("drops the system part and '<|im_start|>user\\n' (19 of 53 tokens kept)", () => {
    expect(COMFYUI.length - templateEnd(COMFYUI)).toBe(19);
    expect(COMFYUI[templateEnd(COMFYUI)]).toBe(64); // "a"
  });
});
