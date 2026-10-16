# Reference outputs

Small Python scripts that compute parts of a model with its official implementation (diffusers, PyTorch),
so the browser engine can be checked against them.

| Script | What it saves |
|---|---|
| `block.py` | Input, timestep modulation, positions and output of one Krea 2 transformer block |

Run with [uv](https://docs.astral.sh/uv/) (no setup needed):

```bash
uv run --with torch --with "diffusers>=0.40" --with safetensors \
    tools/reference/block.py --model path/to/krea2_turbo_fp8_scaled.safetensors
```

The model file stays where it is; only one block is read (about 0.5 GB).

## License

These scripts are test tools, not part of the engine. Each one states its license (GPL-3.0 or Apache-2.0)
in its first line.
