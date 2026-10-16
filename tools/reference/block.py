# SPDX-License-Identifier: Apache-2.0
"""Reference output of one Krea 2 transformer block, computed by diffusers, for the browser engine.

Reads only block N of a ComfyUI Krea 2 file (krea2_turbo_fp8_scaled.safetensors), runs it in float32
on the CPU with diffusers' own Krea2TransformerBlock, and saves the inputs and the output.

    uv run --with torch --with "diffusers>=0.40" --with safetensors \
        tools/reference/block.py --model path/to/krea2_turbo_fp8_scaled.safetensors

Then open the block page and pick both files.
"""

import argparse
import json

import torch
from diffusers.models.transformers.transformer_krea2 import Krea2RotaryPosEmbed, Krea2TransformerBlock
from safetensors import safe_open
from safetensors.torch import save_file

HIDDEN, MLP, HEADS, KV_HEADS, EPS = 6144, 16384, 48, 12, 1e-5
AXES, THETA = [32, 48, 48], 1000.0

# ComfyUI name in the file -> diffusers name in Krea2TransformerBlock
NAMES = {
    "attn.wq.weight": "attn.to_q.weight",
    "attn.wk.weight": "attn.to_k.weight",
    "attn.wv.weight": "attn.to_v.weight",
    "attn.gate.weight": "attn.to_gate.weight",
    "attn.wo.weight": "attn.to_out.0.weight",
    "attn.qknorm.qnorm.scale": "attn.norm_q.weight",
    "attn.qknorm.knorm.scale": "attn.norm_k.weight",
    "mlp.gate.weight": "ff.gate.weight",
    "mlp.up.weight": "ff.up.weight",
    "mlp.down.weight": "ff.down.weight",
    "prenorm.scale": "norm1.weight",
    "postnorm.scale": "norm2.weight",
    "mod.lin": "scale_shift_table",
}


def load_block(path: str, n: int) -> dict[str, torch.Tensor]:
    state = {}
    with safe_open(path, framework="pt") as f:
        keys = set(f.keys())
        for comfy, name in NAMES.items():
            t = f.get_tensor(f"blocks.{n}.{comfy}")
            scale_key = f"blocks.{n}.{comfy}_scale"  # fp8 weights: real value = fp8 x weight_scale
            if t.dtype == torch.float8_e4m3fn:
                t = t.float() * f.get_tensor(scale_key).float()
            state[name] = t.float()
        assert all(f"blocks.{n}.{c}" in keys for c in NAMES)
    state["scale_shift_table"] = state["scale_shift_table"].reshape(6, HIDDEN)
    return state


def positions(text: int, rows: int, cols: int) -> torch.Tensor:
    image = [[0, r, c] for r in range(rows) for c in range(cols)]
    return torch.tensor([[0, 0, 0]] * text + image, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="krea2_turbo_fp8_scaled.safetensors")
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--text", type=int, default=8, help="text tokens")
    parser.add_argument("--rows", type=int, default=4, help="image token rows")
    parser.add_argument("--cols", type=int, default=8, help="image token columns")
    parser.add_argument("--out", default="krea2-block-reference.safetensors")
    args = parser.parse_args()

    state = load_block(args.model, args.block)
    for name in ("norm1.weight", "norm2.weight", "attn.norm_q.weight"):
        w = state[name]
        print(f"{name}: mean {w.mean():.4f}, min {w.min():.4f}, max {w.max():.4f}  (near 0 = stored as w, near 1 = stored as 1 + w)")

    block = Krea2TransformerBlock(HIDDEN, MLP, HEADS, KV_HEADS, EPS).float().eval()
    block.load_state_dict(state, strict=True)

    g = torch.Generator().manual_seed(0)
    tokens = args.text + args.rows * args.cols
    x = torch.randn(1, tokens, HIDDEN, generator=g)
    temb = torch.randn(1, 1, 6 * HIDDEN, generator=g) * 0.5  # stands in for time_mod_proj's output
    ids = positions(args.text, args.rows, args.cols)
    rope = Krea2RotaryPosEmbed(THETA, AXES)(ids)

    with torch.no_grad():
        out = block(x, temb, rope)
    out = out[0] if isinstance(out, tuple) else out

    save_file(
        {"x": x[0].contiguous(), "temb": temb.flatten().contiguous(), "positions": ids, "out": out[0].contiguous()},
        args.out,
        metadata={"block": str(args.block), "shape": json.dumps({"text": args.text, "rows": args.rows, "cols": args.cols})},
    )
    print(f"Saved {args.out}: {tokens} tokens, block {args.block}, output mean {out.mean():.5f}, std {out.std():.5f}")


if __name__ == "__main__":
    main()
