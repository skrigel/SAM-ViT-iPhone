"""Where does an fp16 encoder diverge? Cosine of every block's output vs PyTorch fp32, per compute unit.

Converts the encoder with every tapped module's output as an extra model output (forward hooks during tracing),
runs it on a few dev images per compute unit, and prints/saves the cosine at each tap.

  python tools/taps.py --gelu native --units cpu,gpu,ane
  python tools/taps.py --gelu clenshaw --fine 3,4      # also tap the convs inside each block of stages 3 and 4
"""

import argparse
import json
import os
import re
import sys

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert import EncoderWrapper  # noqa: E402
from eval_coco import COCO, letterbox, split  # noqa: E402
from evitsam import INPUT_SIZE, RESULTS, ROOT, ClenshawGELU, load_sam, make_coreml_friendly, replace_gelu  # noqa: E402

UNITS = {"cpu": ct.ComputeUnit.CPU_ONLY, "gpu": ct.ComputeUnit.CPU_AND_GPU, "ane": ct.ComputeUnit.CPU_AND_NE}


class Tapped(nn.Module):
    def __init__(self, enc, names):
        super().__init__()
        self.enc, self.names, self.buf = enc, names, []
        mods = dict(enc.named_modules())
        for n in names:
            mods[n].register_forward_hook(lambda m, i, o: self.buf.append(o))

    def forward(self, x):
        self.buf = []
        out = self.enc(x)
        return tuple(self.buf) + (out,)


def tap_names(enc, fine):
    names = []
    bb = enc.encoder.backbone
    for k, st in enumerate(bb.stages):
        blocks = st.op_list if hasattr(st, "op_list") else [st]
        for i, b in enumerate(blocks):
            base = f"encoder.backbone.stages.{k}.op_list.{i}" if hasattr(st, "op_list") else f"encoder.backbone.stages.{k}"
            if k in fine:
                for sub, _ in b.named_modules():
                    if re.fullmatch(r"(main\.)?(inverted_conv|depth_conv|point_conv|spatial_conv)|context_module|local_module", sub):
                        names.append(f"{base}.{sub}")
            names.append(base)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="l0")
    ap.add_argument("--gelu", default="native", choices=["native", "clenshaw"])
    ap.add_argument("--student", help="pickled encoder module instead of the stock one")
    ap.add_argument("--fine", default="", help="comma-separated stages to tap inside blocks")
    ap.add_argument("--units", default="cpu,gpu,ane")
    ap.add_argument("--images", type=int, default=4)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--extra", default="", help="comma-separated extra module names to tap")
    args = ap.parse_args()
    fine = [int(v) for v in args.fine.split(",") if v]
    tag = args.tag or f"{args.model}_{args.gelu}" + (f"_fine{''.join(map(str, fine))}" if fine else "")

    sam = load_sam(args.model)
    size = INPUT_SIZE[args.model]
    ref = EncoderWrapper(sam).eval() if not args.student else torch.load(args.student, weights_only=False).eval()
    enc = make_coreml_friendly(torch.load(args.student, weights_only=False) if args.student else EncoderWrapper(load_sam(args.model))).eval()
    if args.gelu == "clenshaw":
        replace_gelu(enc, lambda g: ClenshawGELU(g.approximate))
    names = tap_names(enc, fine) + [n for n in args.extra.split(",") if n]
    model = Tapped(enc, names).eval()
    ref_t = Tapped(ref, names).eval()

    x0 = torch.rand(1, 3, size, size) * 255
    with torch.no_grad():
        model(x0)
    path = os.path.join(ROOT, "models", "taps", f"{tag}.mlpackage")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        traced = torch.jit.trace(model, x0)
        outs = [ct.TensorType(name=f"t{i}") for i in range(len(names) + 1)]
        ml = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=x0.shape)], outputs=outs,
                        compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.iOS18)
        ml.save(path)

    xs = []
    for row in split("dev")[: args.images]:
        img = Image.open(os.path.join(COCO, "val2017", row["file"])).convert("RGB")
        xs.append(letterbox(img, size).transpose(2, 0, 1)[None].astype(np.float32))
    with torch.no_grad():
        refs = [[t.numpy() for t in ref_t(torch.from_numpy(x))] for x in xs]

    labels = names + ["output"]
    table = {"taps": labels}
    for u in args.units.split(","):
        m = ct.models.MLModel(path, compute_units=UNITS[u])
        cos = np.zeros((len(xs), len(labels)))
        for j, x in enumerate(xs):
            out = m.predict({"x": x})
            for i in range(len(labels)):
                a, b = out[f"t{i}"].astype(np.float64).ravel(), refs[j][i].astype(np.float64).ravel()
                cos[j, i] = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        table[u] = cos.mean(0).tolist()
    os.makedirs(os.path.join(RESULTS, "taps"), exist_ok=True)
    json.dump(table, open(os.path.join(RESULTS, "taps", f"{tag}.json"), "w"), indent=1)
    units = args.units.split(",")
    print(f"{'tap':60s}" + "".join(f"{u:>9s}" for u in units))
    for i, n in enumerate(labels):
        print(f"{n.replace('encoder.backbone.', ''):60s}" + "".join(f"{table[u][i]:9.4f}" for u in units))


if __name__ == "__main__":
    main()
