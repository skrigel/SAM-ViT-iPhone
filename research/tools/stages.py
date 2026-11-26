"""Measured per-stage latency: convert encoder prefixes (stem..stage k) and time each on CPU/GPU/ANE.

stage_k latency ~= latency(prefix_k) - latency(prefix_{k-1}). The final prefix is the whole encoder,
so the last delta is the neck + output norm.
Usage: python tools/stages.py [l0]  -> results/stages_<model>.json
"""

import json
import os
import subprocess
import sys

import coremltools as ct
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert import EncoderWrapper  # noqa: E402
from evitsam import INPUT_SIZE, RESULTS, ROOT, load_sam, make_coreml_friendly  # noqa: E402
from efficientvit.models.nn.ops import LiteMLA  # noqa: E402


class Prefix(nn.Module):
    def __init__(self, enc: EncoderWrapper, k: int):
        super().__init__()
        self.enc, self.k = enc, k

    def forward(self, image):
        x = (image - self.enc.mean) / self.enc.std
        for stage in self.enc.encoder.backbone.stages[: self.k + 1]:
            x = stage(x)
        return x


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "l0"
    sam = load_sam(name)
    size = INPUT_SIZE[name]
    enc = make_coreml_friendly(EncoderWrapper(sam)).eval()
    stages = enc.encoder.backbone.stages
    outdir = os.path.join(ROOT, "models", "stages")
    os.makedirs(outdir, exist_ok=True)
    example = torch.rand(1, 3, size, size) * 255
    with torch.no_grad():
        enc(example)

    entries = []
    for k in range(len(stages) + 1):
        full = k == len(stages)
        module = enc if full else Prefix(enc, k)
        label = "neck + norm" if full else ("stem" if k == 0 else f"stage{k}")
        has_attn = (not full) and any(isinstance(m, LiteMLA) for m in stages[k].modules())
        path = os.path.join(outdir, f"{name}_prefix{k}.mlpackage")
        if not os.path.exists(path):
            traced = torch.jit.trace(module.eval(), example)
            ml = ct.convert(traced, inputs=[ct.ImageType(name="image", shape=(1, 3, size, size), color_layout=ct.colorlayout.RGB)],
                            compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.iOS18)
            ml.save(path)
        with torch.no_grad():
            shape = list(module(example).shape)
        row = dict(k=k, label=label, attention=has_attn, out_shape=shape, latency={})
        for u in ["cpu", "gpu", "ane"]:
            r = subprocess.run([os.path.join(ROOT, "bench", "sambench"), "encoder", u, "4", path], capture_output=True, text=True)
            row["latency"][u] = json.loads(r.stdout.strip().splitlines()[-1])["latency_ms"]["p50"]
        entries.append(row)
        print(label, shape, "attn" if has_attn else "", {u: round(v, 2) for u, v in row["latency"].items()}, flush=True)

    for i, e in enumerate(entries):
        e["delta"] = {u: e["latency"][u] - (entries[i - 1]["latency"][u] if i else 0) for u in e["latency"]}
    json.dump(entries, open(os.path.join(RESULTS, f"stages_{name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
