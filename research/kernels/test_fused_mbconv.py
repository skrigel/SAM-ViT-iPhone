"""Correctness of the fused MBConv kernels on real activations, per block (no timing).

For each eligible block: capture its input from COCO dev images (PyTorch fp32 encoder), then compare
  fused  = Core AI GPU, FusedMBConvBlock (fp16)
  stock  = Core AI GPU, the original block exported in fp16 (unfused; Core AI's own kernels)
against the PyTorch fp32 block output. The fused block passes if its error is no worse than ~1.5x the stock fp16 error.

  USE_OS_COREAI=1 .venv-coreai/bin/python kernels/test_fused_mbconv.py [--images 2]  -> results/kernels/mbconv_correctness.json
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
from coreai_kernel_util import compare, run_on_gpu  # noqa: E402
from fused_mbconv import KERNELS, FusedMBConvBlock  # noqa: E402
from eval_coco import COCO, letterbox, split  # noqa: E402
from evitsam import RESULTS, load_sam  # noqa: E402
from convert import EncoderWrapper  # noqa: E402


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=2)
    ap.add_argument("--blocks", default="stages.3.op_list.1,stages.4.op_list.1.local_module")
    args = ap.parse_args()
    enc = EncoderWrapper(load_sam("l0")).eval()
    mods = dict(enc.encoder.backbone.named_modules())
    names = args.blocks.split(",")
    caps = {n: [] for n in names}
    hooks = [mods[n].register_forward_hook(lambda m, i, o, n=n: caps[n].append((i[0].detach().clone(), o.detach().clone())))
             for n in names]
    for row in split("dev")[: args.images]:
        x = letterbox(Image.open(os.path.join(COCO, "val2017", row["file"])).convert("RGB"), 512).transpose(2, 0, 1)[None]
        with torch.no_grad():
            enc(torch.from_numpy(x.astype(np.float32)))
    for h in hooks:  # otherwise exporting the stock block below runs the hook on fake tensors
        h.remove()
    out = {}
    for n in names:
        block = mods[n]
        fused = FusedMBConvBlock(block).eval()
        stock = copy.deepcopy(block).half().eval()
        rows = []
        for j, (xin, yref) in enumerate(caps[n]):
            xh = xin.half()
            (yf,) = run_on_gpu(Wrap(fused), [xh], KERNELS)
            (ys,) = run_on_gpu(Wrap(stock), [xh], [])
            print(f"[{n}] image {j}  shape {tuple(xin.shape)} -> {tuple(yref.shape)}")
            rf = compare("  fused (Core AI GPU, custom kernels)", yf, yref)
            rs = compare("  stock (Core AI GPU, fp16)", ys, yref)
            rows.append(dict(fused=rf, stock=rs))
        ratio = np.mean([r["fused"]["rel_l2"] / r["stock"]["rel_l2"] for r in rows])
        out[n] = dict(rows=rows, rel_l2_ratio_fused_over_stock=float(ratio), passed=bool(ratio < 1.5))
        print(f"[{n}] fused/stock rel-L2 error ratio {ratio:.2f} -> {'PASS' if ratio < 1.5 else 'FAIL'}")
    os.makedirs(os.path.join(RESULTS, "kernels"), exist_ok=True)
    json.dump(out, open(os.path.join(RESULTS, "kernels", "mbconv_correctness.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
