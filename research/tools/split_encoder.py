"""Split the encoder into two Core AI models for a manual GPU + Neural Engine pipeline.

  A (ANE): image -> stem .. stage k, returning every stage output the neck needs that is <= k
  B (GPU): those features -> stages k+1..4 -> neck -> norm -> image_embeddings

Neither Core ML nor Core AI splits this encoder across units by itself (each run places the whole model on one unit),
so this is the "GPU + ANE" baseline the fused kernels are compared against. Stages 0-2 are accurate on the ANE with
the stock GELU (taps: cos >= 0.9997); stage 3 needs the ExpForm GELU there (see PROGRESS.md).

  USE_OS_COREAI=1 .venv-coreai/bin/python tools/split_encoder.py --k 2 [--fuse-mbconv] [--name s2]
  -> work/models/EViTSAM_EncA_<name>.aimodel, EViTSAM_EncB_<name>.aimodel
"""

import argparse
import os
import pathlib
import re
import shutil
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from convert import EncoderWrapper  # noqa: E402
from evitsam import ROOT, ExpFormGELU, load_sam, make_coreml_friendly, replace_gelu  # noqa: E402

NECK_KEYS = ["stage2", "stage3", "stage4"]  # SamNeck.input_keys for L0 (checked at runtime)


class PartA(nn.Module):
    def __init__(self, enc, k, keys):
        super().__init__()
        self.register_buffer("mean", enc.mean.clone())
        self.register_buffer("std", enc.std.clone())
        self.stages = nn.ModuleList(enc.encoder.backbone.stages[: k + 1])
        self.keep = [int(key[5:]) for key in keys if int(key[5:]) <= k]

    def forward(self, image):
        x = (image - self.mean) / self.std
        outs = []
        for i, st in enumerate(self.stages):
            x = st(x)
            if i in self.keep:
                outs.append(x)
        return tuple(outs) if len(outs) > 1 else outs[0]


class PartB(nn.Module):
    def __init__(self, enc, k, keys):
        super().__init__()
        self.k = k
        self.stages = nn.ModuleList(enc.encoder.backbone.stages[k + 1:])
        self.neck, self.norm = enc.encoder.neck, enc.encoder.norm
        self.keys = keys
        self.given = [key for key in keys if int(key[5:]) <= k]

    def forward(self, *feats):
        d = dict(zip(self.given, feats))
        x = feats[-1] if self.k == max(int(g[5:]) for g in self.given) else None
        assert x is not None, "the last given feature must be stage k"
        for i, st in enumerate(self.stages):
            x = st(x)
            d[f"stage{self.k + 1 + i}"] = x
        return self.norm(self.neck(d)["sam_encoder"])


def export(module, example, path, kernels=()):
    from coreai_torch import TorchConverter, get_decomp_table
    ep = torch.export.export(module.eval(), example).run_decompositions(get_decomp_table())
    prog = TorchConverter().register_custom_kernels(list(kernels)).add_exported_program(ep).to_coreai()
    path = pathlib.Path(path)
    shutil.rmtree(path, ignore_errors=True)
    prog.save_asset(path)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=2, help="last stage on the ANE")
    ap.add_argument("--fuse-mbconv", action="store_true", help="fused Metal MBConv kernels in part B")
    ap.add_argument("--expform-a", action="store_true", help="ExpForm GELU in part A's stage 3 (needed if k >= 3)")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    enc = make_coreml_friendly(EncoderWrapper(load_sam("l0"))).eval()
    keys = list(enc.encoder.neck.input_keys)
    assert sorted(keys) == sorted(NECK_KEYS), keys
    keys = sorted(keys)
    x = torch.rand(1, 3, 512, 512) * 255
    with torch.no_grad():
        enc(x)  # warm SeparableResize caches
    a, b = PartA(enc, args.k, keys), PartB(enc, args.k, keys)
    if args.expform_a:
        replace_gelu(a, lambda g: ExpFormGELU(), where=lambda n: re.search(r"stages\.3", n) is not None)
    kernels = []
    if args.fuse_mbconv:
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "kernels"))
        from fused_mbconv import KERNELS, fuse_mbconvs
        print("fused", len(fuse_mbconvs(b)), "MBConv blocks in part B")
        kernels = KERNELS
    with torch.no_grad():
        feats = a(x)
        feats = feats if isinstance(feats, tuple) else (feats,)
        ref = enc(x)
        print("split == whole (fp32, before fusing):", (b(*feats) - ref).abs().max().item() if not args.fuse_mbconv else "skipped")
    a, b = a.half(), b.half()
    xh = x.half()
    with torch.no_grad():
        fh = a(xh)
        fh = fh if isinstance(fh, tuple) else (fh,)
    export(a, (xh,), os.path.join(ROOT, "models", f"EViTSAM_EncA_{args.name}.aimodel"))
    export(b, tuple(f.half() for f in fh), os.path.join(ROOT, "models", f"EViTSAM_EncB_{args.name}.aimodel"), kernels)


if __name__ == "__main__":
    main()
