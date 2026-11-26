"""Convert the EfficientViT-SAM image encoder to Core AI (.aimodel) with coreai-torch.

Runs in the Core AI environment (torch 2.13 + coreai-torch; see init.sh):
  USE_OS_COREAI=1 ../.venv-coreai/bin/python tools/convert_coreai.py --precision fp16 [--anefix] [--student X.pt]

Input `image`: float (1, 3, S, S) RGB pixels in [0, 255] (same contract as the Core ML encoder); output: (1, 256, 64, 64).
fp16 is produced by exporting the module in half precision (Core AI keeps the exported dtypes).
"""

import argparse
import os
import pathlib
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert import EncoderWrapper  # noqa: E402
from evitsam import INPUT_SIZE, ROOT, ClenshawGELU, load_sam, make_coreml_friendly, replace_gelu  # noqa: E402


def build(model, anefix, student=None):
    enc = torch.load(student, weights_only=False) if student else EncoderWrapper(load_sam(model))
    # Resize-as-matmul + fp16-safe attention (exact rewrites; the stock bicubic interpolate does not export in fp16).
    enc = make_coreml_friendly(enc)
    if anefix:
        replace_gelu(enc, lambda g: ClenshawGELU(g.approximate))
    return enc.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="l0")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--anefix", action="store_true", help="ANE-accurate GELU + linear resize (evitsam.py)")
    ap.add_argument("--student")
    ap.add_argument("--name")
    args = ap.parse_args()

    from coreai_torch import TorchConverter, get_decomp_table

    size = INPUT_SIZE[args.model]
    enc = build(args.model, args.anefix, args.student)
    x = torch.rand(1, 3, size, size) * 255
    with torch.no_grad():
        enc(x)  # warm SeparableResize caches so export sees constants
    if args.precision == "fp16":
        enc, x = enc.half(), x.half()

    class Named(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, image):
            return self.m(image)

    ep = torch.export.export(Named(enc).eval(), (x,)).run_decompositions(get_decomp_table())
    prog = TorchConverter().add_exported_program(ep).to_coreai()
    name = args.name or f"{args.model}_{args.precision}" + ("_anefix" if args.anefix else "")
    out = pathlib.Path(ROOT) / "models" / f"EViTSAM_Encoder_{name}.aimodel"
    shutil.rmtree(out, ignore_errors=True)
    prog.save_asset(out)
    print("saved", out)


if __name__ == "__main__":
    main()
