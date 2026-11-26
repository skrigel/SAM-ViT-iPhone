"""Numerical check: Core ML (fp16/fp32, each compute unit) vs PyTorch fp32 on real images with box prompts.

Usage: python tools/accuracy.py [l0 l2 xl0]  -> results/accuracy.json (merged; saved after every case)
"""

import json
import os
import sys

import coremltools as ct
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert import DecoderWrapper, EncoderWrapper  # noqa: E402
from evitsam import INPUT_SIZE, RESULTS, ROOT, load_sam  # noqa: E402

FIG = os.path.join(ROOT, "third_party", "efficientvit", "assets", "fig")
# (image, box in original pixel coords xyxy) — boxes picked around the main object of each demo image.
CASES = [("cat.jpg", None), ("bear.jpeg", None), ("girl.png", None)]
UNITS = {"cpu": ct.ComputeUnit.CPU_ONLY, "gpu": ct.ComputeUnit.CPU_AND_GPU, "ane": ct.ComputeUnit.CPU_AND_NE}


def letterbox(img, size):
    w, h = img.size
    s = size / max(w, h)
    nw, nh = int(w * s + 0.5), int(h * s + 0.5)
    canvas = Image.new("RGB", (size, size))
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (0, 0))
    return canvas, 1024 / max(w, h)


def iou(a, b):
    return float((a & b).sum() / max(1, (a | b).sum()))


def main():
    path = os.path.join(RESULTS, "accuracy.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    for name in sys.argv[1:] or ["l0", "l2", "xl0"]:
        sam = load_sam(name)
        enc_t, dec_t = EncoderWrapper(sam).eval(), DecoderWrapper(sam).eval()
        size = INPUT_SIZE[name]
        for prec in ["fp16", "fp32"]:
            for uname, u in UNITS.items():
                if prec == "fp32" and uname == "ane":
                    continue  # the Neural Engine only computes in fp16
                if name == "xl0" and uname == "ane":
                    continue  # does not compile for the ANE; falls back to a ~1.3 s/frame CPU path
                enc = ct.models.MLModel(os.path.join(ROOT, "models", f"EViTSAM_Encoder_{name}_{prec}.mlpackage"), compute_units=u)
                dec = ct.models.MLModel(os.path.join(ROOT, "models", f"EViTSAM_Decoder_{name}_{prec}.mlpackage"), compute_units=u)
                rows = []
                for fname, _ in CASES:
                    img = Image.open(os.path.join(FIG, fname)).convert("RGB")
                    w, h = img.size
                    box = np.array([[[w * 0.2, h * 0.15], [w * 0.8, h * 0.9]]], dtype=np.float32)
                    lb, s = letterbox(img, size)
                    with torch.no_grad():
                        x = torch.from_numpy(np.asarray(lb)).permute(2, 0, 1)[None].float()
                        emb_ref = enc_t(x)
                        box_t = torch.from_numpy(box * s)
                        m_ref, iou_ref = dec_t(emb_ref, box_t, torch.zeros(1, 1, 256, 256), torch.zeros(1))
                    emb = enc.predict({"image": lb})["image_embeddings"]
                    d = dec.predict({"image_embeddings": emb.astype(np.float32), "box_coords": box * s,
                                     "mask_input": np.zeros((1, 1, 256, 256), np.float32), "has_mask_input": np.zeros(1, np.float32)})
                    er = emb_ref.numpy().ravel()
                    cos = float(np.dot(emb.ravel(), er) / (np.linalg.norm(emb) * np.linalg.norm(er)))
                    rows.append(dict(
                        image=fname,
                        emb_cosine=cos,
                        emb_max_abs_err=float(np.abs(emb.ravel() - er).max()),
                        emb_has_nan=bool(np.isnan(emb).any()),
                        mask_iou_vs_torch=iou(d["low_res_masks"] > 0, m_ref.numpy() > 0),
                        pred_iou=float(np.asarray(d["iou"]).ravel()[0]),
                        pred_iou_ref=float(iou_ref.numpy().ravel()[0]),
                    ))
                key = f"{name}_{prec}_{uname}"
                out[key] = rows
                if name == "l0":
                    # Decoder alone, fed the exact fp32 PyTorch embedding, so its own numerics are isolated.
                    drows = []
                    for fname, _ in CASES:
                        img = Image.open(os.path.join(FIG, fname)).convert("RGB")
                        w, h = img.size
                        box = np.array([[[w * 0.2, h * 0.15], [w * 0.8, h * 0.9]]], dtype=np.float32)
                        lb, s = letterbox(img, size)
                        with torch.no_grad():
                            emb_ref = enc_t(torch.from_numpy(np.asarray(lb)).permute(2, 0, 1)[None].float())
                            m_ref, _ = dec_t(emb_ref, torch.from_numpy(box * s), torch.zeros(1, 1, 256, 256), torch.zeros(1))
                        d = dec.predict({"image_embeddings": emb_ref.numpy(), "box_coords": box * s,
                                         "mask_input": np.zeros((1, 1, 256, 256), np.float32), "has_mask_input": np.zeros(1, np.float32)})
                        drows.append(dict(image=fname, mask_iou_vs_torch=iou(d["low_res_masks"] > 0, m_ref.numpy() > 0)))
                    out[f"decoder_{prec}_{uname}"] = drows
                    print(f"decoder_{prec}_{uname}", " ".join(f"{r['image']}: maskIoU={r['mask_iou_vs_torch']:.4f}" for r in drows), flush=True)
                print(key, " ".join(f"{r['image']}: cos={r['emb_cosine']:.5f} maskIoU={r['mask_iou_vs_torch']:.4f}"
                                    + (" NaN!" if r["emb_has_nan"] else "") for r in rows), flush=True)
                json.dump(out, open(path, "w"), indent=1)  # incremental: Core ML has crashed the process mid-run


if __name__ == "__main__":
    main()
