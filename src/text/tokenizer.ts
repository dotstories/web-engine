/**
 * Qwen2 tokenizer (byte-level BPE), as used by Qwen3-VL and Krea 2's text encoder.
 * Same steps as Hugging Face's Qwen2Tokenizer: NFC, split out special tokens, split the rest with
 * Qwen2's pre-tokenizer pattern, map bytes to printable characters (GPT-2 style), merge with BPE.
 */

/** Qwen2's pre-tokenizer pattern; `(?i:'s|…)` written out, as JavaScript has no inline flags. */
const PATTERN =
  /'(?:[sStTmMdD]|[rR][eE]|[vV][eE]|[lL][lL])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

/** Special tokens of the Qwen2/Qwen3 vocabulary (added tokens, never split by BPE). */
export const SPECIAL_TOKENS: Record<string, number> = {
  "<|endoftext|>": 151643,
  "<|im_start|>": 151644,
  "<|im_end|>": 151645,
  "<|object_ref_start|>": 151646,
  "<|object_ref_end|>": 151647,
  "<|box_start|>": 151648,
  "<|box_end|>": 151649,
  "<|quad_start|>": 151650,
  "<|quad_end|>": 151651,
  "<|vision_start|>": 151652,
  "<|vision_end|>": 151653,
  "<|vision_pad|>": 151654,
  "<|image_pad|>": 151655,
  "<|video_pad|>": 151656,
  "<tool_call>": 151657,
  "</tool_call>": 151658,
  "<|fim_prefix|>": 151659,
  "<|fim_middle|>": 151660,
  "<|fim_suffix|>": 151661,
  "<|fim_pad|>": 151662,
  "<|repo_name|>": 151663,
  "<|file_sep|>": 151664,
  "<tool_response>": 151665,
  "</tool_response>": 151666,
  "<think>": 151667,
  "</think>": 151668,
};

/** GPT-2's byte → printable character map (every byte gets a visible, unique character). */
function byteChars(): string[] {
  const bytes: number[] = [];
  for (let b = 33; b <= 126; b++) bytes.push(b);
  for (let b = 161; b <= 172; b++) bytes.push(b);
  for (let b = 174; b <= 255; b++) bytes.push(b);
  const chars = bytes.map((b) => String.fromCharCode(b));
  let extra = 0;
  const out: string[] = [];
  for (let b = 0; b < 256; b++) out[b] = bytes.includes(b) ? chars[bytes.indexOf(b)] : String.fromCharCode(256 + extra++);
  return out;
}

const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

export class Qwen2Tokenizer {
  private readonly ranks = new Map<string, number>();
  private readonly cache = new Map<string, number[]>();
  private readonly bytes = byteChars();
  private readonly encoder = new TextEncoder();
  private readonly special: RegExp;
  private readonly vocab: Record<string, number>;
  private readonly specialTokens: Record<string, number>;

  constructor(vocab: Record<string, number>, merges: string, specialTokens: Record<string, number> = SPECIAL_TOKENS) {
    this.vocab = vocab;
    this.specialTokens = specialTokens;
    merges.split("\n").forEach((line, i) => {
      if (line && !line.startsWith("#version")) this.ranks.set(line, i);
    });
    this.special = new RegExp(`(${Object.keys(specialTokens).map(escape).join("|")})`);
  }

  /** Loads vocab.json and merges.txt from URLs (or anything fetch() reads). */
  static async load(vocabUrl: string | URL, mergesUrl: string | URL): Promise<Qwen2Tokenizer> {
    const [vocab, merges] = await Promise.all([fetch(vocabUrl).then((r) => r.json()), fetch(mergesUrl).then((r) => r.text())]);
    return new Qwen2Tokenizer(vocab, merges);
  }

  encode(text: string): number[] {
    const ids: number[] = [];
    for (const part of text.normalize("NFC").split(this.special)) {
      if (!part) continue;
      if (part in this.specialTokens) {
        ids.push(this.specialTokens[part]);
        continue;
      }
      for (const piece of part.match(PATTERN) ?? []) ids.push(...this.bpe(piece));
    }
    return ids;
  }

  private bpe(piece: string): number[] {
    const cached = this.cache.get(piece);
    if (cached) return cached;
    let word = Array.from(this.encoder.encode(piece), (b) => this.bytes[b]);
    for (;;) {
      let best = -1;
      let bestRank = Infinity;
      for (let i = 0; i < word.length - 1; i++) {
        const rank = this.ranks.get(`${word[i]} ${word[i + 1]}`);
        if (rank !== undefined && rank < bestRank) [best, bestRank] = [i, rank];
      }
      if (best < 0) break;
      const [a, b] = [word[best], word[best + 1]];
      const merged: string[] = [];
      for (let i = 0; i < word.length; i++) {
        if (i < word.length - 1 && word[i] === a && word[i + 1] === b) {
          merged.push(a + b);
          i++;
        } else merged.push(word[i]);
      }
      word = merged;
    }
    const ids = word.map((token) => {
      const id = this.vocab[token];
      if (id === undefined) throw new Error(`Token not in the vocabulary: ${JSON.stringify(token)}`);
      return id;
    });
    this.cache.set(piece, ids);
    return ids;
  }
}
