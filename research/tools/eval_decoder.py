"""Check the app's prompt decoder (iPhone-SAM-ViT/Models/SAMPromptDecoder.mlpackage) per compute unit.

Feeds the PyTorch fp32 embedding (so only the decoder varies) and the GT box as the app sends it (two corners,
labels 2 and 3, 1024 frame). Scores mIoU vs GT for two mask-selection rules:
  mask0  - the single-mask output (what SAM recommends for box prompts)
  argmax - highest predicted IoU of the 4 outputs (what Segmenter.decode does today)

  python tools/eval_decoder.py [--units cpu,gpu,ane] [--n 200]   -> results/decoder_<unit>.json
"""

import argparse
import json
import os
import sys

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert import DecoderWrapper, EncoderWrapper  # noqa: E402
from eval_coco import COCO, gt_mask, letterbox, split  # noqa: E402
from evitsam import RESEARCH, RESULTS, load_sam  # noqa: E402

APP_DECODER = os.path.join(os.path.dirname(RESEARCH), "iPhone-SAM-ViT", "Models", "SAMPromptDecoder.mlpackage")
UNITS = {"cpu": ct.ComputeUnit.CPU_ONLY, "gpu": ct.ComputeUnit.CPU_AND_GPU, "ane": ct.ComputeUnit.CPU_AND_NE, "all": ct.ComputeUnit.ALL}


def to_mask(low, row):
    s = 1024 / max(row["w"], row["h"])
    m = F.interpolate(torch.from_numpy(low.astype(np.float32))[None, None], (1024, 1024), mode="bilinear", align_corners=False)
    ih, iw = int(row["h"] * s + 0.5), int(row["w"] * s + 0.5)
    m = F.interpolate(m[..., :ih, :iw], (row["h"], row["w"]), mode="bilinear", align_corners=False)
    return (m[0, 0] > 0).numpy()


def iou(a, b):
    return (a & b).sum() / max(1, (a | b).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default="cpu,gpu,ane")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--decoder", default=APP_DECODER)
    args = ap.parse_args()

    sam = load_sam("l0")
    enc = EncoderWrapper(sam).eval().to("mps")
    dec_t = DecoderWrapper(sam).eval()
    rows = split("dev")[: args.n]
    cache = []
    for row in rows:
        img = Image.open(os.path.join(COCO, "val2017", row["file"])).convert("RGB")
        x = letterbox(img, 512).transpose(2, 0, 1)[None].astype(np.float32)
        with torch.no_grad():
            emb = enc(torch.from_numpy(x).to("mps")).cpu()
        s = 1024 / max(row["w"], row["h"])
        bx, by, bw, bh = row["bbox"]
        box = np.array([[[bx * s, by * s], [(bx + bw) * s, (by + bh) * s]]], np.float32)
        with torch.no_grad():
            low, _ = dec_t(emb, torch.from_numpy(box), torch.zeros(1, 1, 256, 256), torch.zeros(1))
        cache.append((row, emb.numpy(), box, low[0, 0].numpy()))

    ref = [iou(to_mask(low, row), gt_mask(row)) for row, _, _, low in cache]
    print(f"torch fp32 decoder (mask0): mIoU {100 * np.mean(ref):.2f}")
    for u in args.units.split(","):
        m = ct.models.MLModel(args.decoder, compute_units=UNITS[u])
        r0, ra, agree = [], [], []
        for (row, emb, box, low_ref), r in zip(cache, ref):
            out = m.predict({"image_embeddings": emb.astype(np.float16), "point_coords": box.astype(np.float16),
                             "point_labels": np.array([[2, 3]], np.float16)})
            masks, ious = out["masks"][0].astype(np.float32), out["iou_predictions"][0].astype(np.float32)
            gt = gt_mask(row)
            r0.append(iou(to_mask(masks[0], row), gt))
            ra.append(iou(to_mask(masks[int(np.argmax(ious))], row), gt))
            agree.append(iou(masks[0] > 0, low_ref > 0))
        res = dict(unit=u, n=len(cache), ref_mIoU=100 * float(np.mean(ref)), mask0_mIoU=100 * float(np.mean(r0)),
                   argmax_mIoU=100 * float(np.mean(ra)), mask0_vs_torch_iou=float(np.mean(agree)),
                   mask0_vs_torch_min=float(np.min(agree)))
        json.dump(res, open(os.path.join(RESULTS, f"decoder_{u}.json"), "w"), indent=1)
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
