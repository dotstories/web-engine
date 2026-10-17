/**
 * The Weeping Gate: the Dark Fantasy Kit image the browser must reproduce.
 * Made in ComfyUI (Krea 2 Turbo, Dark Fantasy Kit workflow) with these settings. The Kit's "Refine prompt"
 * step had rewritten the scene + style first; prompt.txt is that rewritten text, byte for byte, as the
 * text encoder got it (ComfyUI history, "Preview as Text" node). The LoRA was off.
 */
export const WEEPING_GATE = {
  prompt: "../../dots/weeping-gate/prompt.txt",
  reference: "../../dots/weeping-gate/reference.webp",
  promptSha256: "6a411ed9bb3ab29dac8d903cfffd3798d476568eabee0c8bf851a8fe3c148d87",
  width: 1928, // ResolutionSelector 16:9, 2 MP, multiple 8
  height: 1088,
  seed: 54460995,
  loraStrength: 0,
} as const;
