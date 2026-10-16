# SPDX-License-Identifier: GPL-3.0-only
"""Reference output of the Krea 2 text encoder (Qwen3-VL 4B, 12 tapped layers), computed by ComfyUI's code.

Tokenizes a prompt with ComfyUI's Krea 2 tokenizer and template, encodes it on the CPU in float32 and saves
the token ids and the conditioning the transformer receives: [tokens, 12 · 2560] (the template prefix
already stripped, like ComfyUI does). Needs about 20 GB of RAM.

    & "<ComfyUI install>\\.venv\\Scripts\\python.exe" tools/reference/text.py --te path/to/qwen3vl_4b_fp8_scaled.safetensors
"""

import argparse
import sys

from transformer import find_comfyui  # same folder

PROMPT = "a dark fantasy castle on a cliff at night, lightning, ravens"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--te", required=True, help="qwen3vl_4b_fp8_scaled.safetensors")
    parser.add_argument("--comfyui", help="ComfyUI folder (contains comfy/)")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--out", default="krea2-text-reference.safetensors")
    args = parser.parse_args()

    sys.path.insert(0, str(find_comfyui(args.comfyui)))
    import comfy.cli_args

    comfy.cli_args.args.cpu = True  # float32 on the CPU: the most exact reference
    comfy.cli_args.args.force_fp32 = True

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    import comfy.sd

    with safe_open(args.te, framework="pt") as f:
        keys = sorted(f.keys())
        print(f"{len(keys)} tensors, for example:")
        for k in [k for k in keys if ".layers.0." in k or "layers" not in k][:40]:
            print(f"  {k}: {f.get_slice(k).get_dtype()} {f.get_slice(k).get_shape()}")

    clip = comfy.sd.load_clip([args.te], clip_type=comfy.sd.CLIPType.KREA2)
    tokens = clip.tokenize(args.prompt)
    ids = [int(t[0]) for t in tokens["qwen3vl_4b"][0]]
    print(f"{len(ids)} tokens with the template: {ids}")

    with torch.no_grad():
        cond = clip.encode_from_tokens(tokens)  # [1, tokens after the template prefix, 12 · 2560]
    cond = cond[0].float()
    print(f"conditioning {list(cond.shape)} mean {cond.mean():.5f} std {cond.std():.5f}")

    save_file(
        {"ids": torch.tensor(ids, dtype=torch.float32), "cond": cond.contiguous()},
        args.out,
        metadata={"prompt": args.prompt, "kept": str(cond.shape[0])},
    )
    print(f"Saved {args.out}: kept the last {cond.shape[0]} of {len(ids)} tokens")


if __name__ == "__main__":
    main()
