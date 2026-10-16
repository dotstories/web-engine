/**
 * Krea 2's prompt template (ComfyUI comfy/text_encoders/krea2.py, the same system prompt as Qwen-Image),
 * without the empty <think> block. The text encoder sees the whole template; the transformer only gets
 * the tokens after "<|im_start|>user\n".
 */
export const KREA2_TEMPLATE =
  "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n";

export const kreaPrompt = (prompt: string) => KREA2_TEMPLATE.replace("{}", prompt);

const IM_START = 151644;
const USER = 872;
const NEWLINE = 198;

/** Index of the first token after the second "<|im_start|>" + "user\n" (ComfyUI's template_end). */
export function templateEnd(ids: number[]): number {
  let end = -1;
  let starts = 0;
  for (let i = 0; i < ids.length && starts < 2; i++) {
    if (ids[i] === IM_START) {
      end = i;
      starts++;
    }
  }
  if (end >= 0 && ids.length > end + 3 && ids[end + 1] === USER && ids[end + 2] === NEWLINE) end += 3;
  return Math.max(end, 0);
}
