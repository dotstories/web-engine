# SPDX-License-Identifier: GPL-3.0-only
"""Reference output of the Krea 2 image decoder (the Wan 2.1 / Qwen-Image VAE), computed by ComfyUI's code.

Decodes a seeded random latent with ComfyUI's WanVAE and saves the latent, the image and the output of
each decoder stage. Runs on the CPU in float32.

    & "<ComfyUI install>\\.venv\\Scripts\\python.exe" tools/reference/vae.py --vae path/to/qwen_image_vae.safetensors
"""

import argparse
import sys

from transformer import find_comfyui  # same folder


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vae", required=True, help="qwen_image_vae.safetensors")
    parser.add_argument("--comfyui", help="ComfyUI folder (contains comfy/)")
    parser.add_argument("--latent", type=int, default=16, help="latent height and width (image = 8x)")
    parser.add_argument("--out", default="krea2-vae-reference.safetensors")
    args = parser.parse_args()

    sys.path.insert(0, str(find_comfyui(args.comfyui)))
    import torch
    from safetensors.torch import load_file, save_file

    from comfy.ldm.wan.vae import WanVAE

    state = {k: v.float() for k, v in load_file(args.vae).items()}
    decoder_keys = sorted(k for k in state if k.startswith(("decoder.", "conv2.")))
    for k in decoder_keys:
        print(f"  {k}: {list(state[k].shape)}")
    dim = state["decoder.head.0.gamma"].shape[0]
    z_dim = state["conv2.weight"].shape[0]
    print(f"dim {dim}, z_dim {z_dim}, {len(state)} tensors")

    # Which up stages have a temporal conv differs between VAEs (Qwen-Image: the first two, Wan 2.1: the last two).
    # It doesn't change a single image (those convs are skipped for one frame), but loading needs the right layout.
    resamples = sorted({int(k.split(".")[2]) for k in state if k.startswith("decoder.upsamples.") and ".resample." in k})
    upsample = [f"decoder.upsamples.{i}.time_conv.weight" in state for i in resamples]
    print(f"temporal upsample per stage: {upsample}")
    vae = WanVAE(dim=dim, z_dim=z_dim, temperal_downsample=upsample[::-1]).eval()
    vae.load_state_dict(state, strict=True)

    saved = {}
    hook = lambda name: lambda _m, _i, out: saved.__setitem__(name, out.detach())  # noqa: E731
    vae.conv2.register_forward_hook(hook("conv2"))
    vae.decoder.conv1.register_forward_hook(hook("conv1"))
    vae.decoder.middle.register_forward_hook(hook("middle"))
    for i, layer in enumerate(vae.decoder.upsamples):
        layer.register_forward_hook(hook(f"up{i}"))

    g = torch.Generator().manual_seed(0)
    z = torch.randn(1, z_dim, 1, args.latent, args.latent, generator=g)
    with torch.no_grad():
        image = vae.decode(z)

    # channel-last [H·W, C] per stage, to compare with the browser engine
    tensors = {"latent": z[0, :, 0], "image": image[0, :, 0]}
    tensors |= {f"step.{k}": v[0, :, 0].permute(1, 2, 0).reshape(-1, v.shape[1]) for k, v in saved.items()}
    save_file({k: v.contiguous().float() for k, v in tensors.items()}, args.out, metadata={"latent": str(args.latent)})
    print(f"Saved {args.out}")
    for k, v in tensors.items():
        print(f"  {k}: {list(v.shape)} mean {v.mean():.5f} std {v.std():.5f}")


if __name__ == "__main__":
    main()
