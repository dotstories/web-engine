# Reference outputs

Small Python scripts that compute parts of a model with its official implementation (diffusers, PyTorch),
so the browser engine can be checked against them.

| Script | What it saves |
|---|---|
| `block.py` | Input, timestep modulation, positions and output of one Krea 2 transformer block (diffusers) |
| `text.py` | The text encoder (Qwen3-VL 4B, ComfyUI's Krea 2 tokenizer and template): token ids and the transformer's conditioning |
| `vae.py` | The image decoder (Wan 2.1 / Qwen-Image VAE, ComfyUI's code): latent, image and every decoder stage |
| `transformer.py` | Krea 2 with only its first N blocks (ComfyUI's code): inputs, output and every step in between |

Run with [uv](https://docs.astral.sh/uv/) (no setup needed):

```bash
uv run --with torch --with "diffusers>=0.40" --with safetensors \
    tools/reference/block.py --model path/to/krea2_turbo_fp8_scaled.safetensors
```

The model file stays where it is; only one block is read (about 0.5 GB).

`transformer.py` uses ComfyUI's own model code, so run it with the Python of your ComfyUI install
(Comfy Desktop: `%LOCALAPPDATA%\Comfy-Desktop\ComfyUI-Installs\<install>\.venv\Scripts\python.exe`).

## License

These scripts are test tools, not part of the engine. Each one states its license (GPL-3.0 or Apache-2.0)
in its first line.
