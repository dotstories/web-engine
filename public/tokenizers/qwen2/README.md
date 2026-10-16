# Qwen2 tokenizer files

`vocab.json` and `merges.txt` are the byte-level BPE tokenizer of the Qwen2 / Qwen2.5 / Qwen3 models by the
Qwen team (Alibaba Cloud), licensed under the [Apache License 2.0](LICENSE). They are the same files that
Hugging Face's `Qwen/Qwen2.5-*` repositories and ComfyUI (`comfy/text_encoders/qwen25_tokenizer`) use.

The DotStories Web Engine uses them to turn prompts into tokens for Krea 2's text encoder (Qwen3-VL 4B).
They are unchanged.
