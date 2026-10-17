# SPDX-License-Identifier: Apache-2.0
"""Checks tests/fixtures/torch-randn.json against real torch (the noise ComfyUI starts every image from).

ComfyUI makes its noise with torch.manual_seed(seed) and torch.randn(shape) on the CPU in float32
(comfy/sample.py prepare_noise). src/model/noise.ts must give the same numbers. Exits 1 on a mismatch.

    pip install torch --index-url https://download.pytorch.org/whl/cpu
    python tools/reference/noise.py tests/fixtures/torch-randn.json
"""

import json
import sys

import torch

TOLERANCE = 1e-5  # float32 log/sin/cos may differ by an ulp between builds


def main(path: str) -> int:
    cases = json.load(open(path, encoding="utf-8"))["cases"]
    bad = 0
    for case in cases:
        torch.manual_seed(case["seed"])
        noise = torch.randn(case["n"], dtype=torch.float32).tolist()
        for index, expected in case["values"].items():
            actual = noise[int(index)]
            if abs(actual - expected) > TOLERANCE * max(1.0, abs(actual)):
                bad += 1
                print(f"n={case['n']} seed={case['seed']} [{index}]: torch {actual!r}, fixture {expected!r}")
    print(f"torch {torch.__version__}: {len(cases)} cases, {bad} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
