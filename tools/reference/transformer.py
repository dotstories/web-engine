# SPDX-License-Identifier: GPL-3.0-only
"""Reference output of Krea 2 with only its first N blocks, computed by ComfyUI's own model code.

The browser engine runs the same small model and compares every step: timestep embedding, text
fusion, image input, blocks and final layer. Runs on the CPU in float32 and needs about 2 GB of RAM.

Run it with the Python of your ComfyUI install (it needs ComfyUI's code and packages), e.g. Comfy Desktop:

    & "$env:LOCALAPPDATA\\Comfy-Desktop\\ComfyUI-Installs\\<install>\\.venv\\Scripts\\python.exe" `
        tools/reference/transformer.py --model path/to/krea2_turbo_fp8_scaled.safetensors

The ComfyUI folder is found automatically (or pass --comfyui).
"""

import argparse
import os
import sys
from pathlib import Path

KREA = "comfy/ldm/krea2/model.py"


def find_comfyui(given: str | None) -> Path:
    candidates = [Path(given)] if given else []
    for parent in Path(sys.executable).resolve().parents:  # the Python of an install sits inside it
        candidates += [parent, parent / "ComfyUI"]
    local = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "Comfy-Desktop"
    candidates += sorted(local.glob("**/ComfyUI")) if local.is_dir() else []
    for c in candidates:
        if (c / KREA).is_file():
            return c
    sys.exit("ComfyUI with Krea 2 support not found. Pass --comfyui <folder that contains comfy/>.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="krea2_turbo_fp8_scaled.safetensors")
    parser.add_argument("--comfyui", help="ComfyUI folder (contains comfy/)")
    parser.add_argument("--blocks", type=int, default=2, help="how many of the 28 blocks to keep")
    parser.add_argument("--text", type=int, default=8, help="text tokens")
    parser.add_argument("--latent", type=int, default=8, help="latent height and width (pixels / 8)")
    parser.add_argument("--timestep", type=float, default=0.7)
    parser.add_argument("--out", default="krea2-transformer-reference.safetensors")
    args = parser.parse_args()

    comfy_dir = find_comfyui(args.comfyui)
    sys.path.insert(0, str(comfy_dir))
    print(f"ComfyUI: {comfy_dir}")

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    import comfy.ops
    from comfy.ldm.krea2.model import SingleStreamDiT

    # Every tensor outside the blocks, plus the first N blocks. fp8 weights: real value = fp8 x weight_scale.
    state = {}
    with safe_open(args.model, framework="pt") as f:
        keys = [k for k in f.keys() if not k.startswith("blocks.") or int(k.split(".")[1]) < args.blocks]
        for key in sorted(keys):
            if key.endswith("_scale"):
                continue
            t = f.get_tensor(key)
            if t.dtype == torch.float8_e4m3fn:
                t = t.float() * f.get_tensor(f"{key}_scale").float()
            state[key] = t.float()
            if not key.startswith("blocks."):
                print(f"  {key}: {f.get_slice(key).get_dtype()} {list(t.shape)}")

    model = SingleStreamDiT(layers=args.blocks, device="cpu", dtype=torch.float32, operations=comfy.ops.disable_weight_init)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Save what goes in and out of each part.
    saved = {}
    for name in ("tmlp", "tproj", "txtfusion", "txtmlp", "first", "last"):
        getattr(model, name).register_forward_hook(lambda _m, _i, out, name=name: saved.__setitem__(name, out.detach()))
    model.blocks[-1].register_forward_hook(lambda _m, _i, out: saved.__setitem__("blocks", out.detach()))

    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 16, args.latent, args.latent, generator=g)
    context = torch.randn(1, args.text, 12 * 2560, generator=g)
    timesteps = torch.tensor([args.timestep])

    with torch.no_grad():
        out = model(x, timesteps, context, transformer_options={})

    tensors = {"x": x[0], "context": context[0], "timestep": timesteps, "out": out[0]}
    tensors |= {f"step.{k}": v.reshape(-1, v.shape[-1]) for k, v in saved.items()}
    save_file({k: v.contiguous().float() for k, v in tensors.items()}, args.out, metadata={
        "blocks": str(args.blocks), "text": str(args.text), "latent": str(args.latent), "timestep": str(args.timestep)})
    print(f"Saved {args.out}: {args.blocks} blocks, {args.text} text tokens, latent {args.latent}x{args.latent}")
    for k, v in tensors.items():
        print(f"  {k}: {list(v.shape)} mean {v.mean():.5f} std {v.std():.5f}")


if __name__ == "__main__":
    main()
