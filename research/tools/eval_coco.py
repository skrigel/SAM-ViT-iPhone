"""Fixed evaluator: box-prompted mIoU on a 500-image COCO val2017 subset.  FROZEN — see PROGRAM.md.

The encoder under test produces the image embedding; the prompt encoder + mask decoder always run in PyTorch fp32,
so the score isolates the encoder. For every image it also reports the cosine between the embedding and the
PyTorch fp32 reference embedding.

  python tools/eval_coco.py --backend torch                                   # fp32 reference
  python tools/eval_coco.py --backend coreml --encoder work/models/X.mlpackage --unit ane
  USE_OS_COREAI=1 .venv-coreai/bin/python tools/eval_coco.py --backend coreai --encoder work/models/X.aimodel --unit gpu
  python tools/eval_coco.py --backend torch --student work/students/relu.pt   # a fine-tuned encoder (whole nn.Module)

Encoders take RGB pixels in [0, 255], shape (1, 3, S, S), letterboxed at the top-left with the mean colour as padding,
and normalize inside the graph (see convert.EncoderWrapper).
"""

import argparse
import asyncio
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evitsam import INPUT_SIZE, RESULTS, ROOT, load_sam  # noqa: E402

COCO = os.path.join(ROOT, "data", "coco")
MEAN_RGB = (124, 116, 104)  # ≈ normalization mean, so the padding is ≈0 after normalization, as in SamPad
SUBSET = os.path.join(RESULTS, "coco_subset.json")


# ---------------------------------------------------------------- data

def build_subset():
    """Seed-0 shuffle of val2017 images; per image the first non-crowd annotation covering >= 1% of it."""
    if os.path.exists(SUBSET):
        return json.load(open(SUBSET))
    ann = json.load(open(os.path.join(COCO, "annotations", "instances_val2017.json")))
    imgs = {im["id"]: im for im in ann["images"]}
    by_img = {}
    for a in sorted(ann["annotations"], key=lambda a: a["id"]):
        im = imgs[a["image_id"]]
        if a["iscrowd"] or a["area"] < 0.01 * im["width"] * im["height"] or a["image_id"] in by_img:
            continue
        by_img[a["image_id"]] = a
    ids = sorted(by_img)
    np.random.default_rng(0).shuffle(ids)
    rows = [dict(file=imgs[i]["file_name"], w=imgs[i]["width"], h=imgs[i]["height"], bbox=by_img[i]["bbox"],
                 segmentation=by_img[i]["segmentation"], category_id=by_img[i]["category_id"]) for i in ids[:1000]]
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(rows, open(SUBSET, "w"))
    return rows


def split(name):
    rows = build_subset()
    return rows[:500] if name == "eval" else rows[500:1000]


def gt_mask(row):
    m = np.zeros((row["h"], row["w"]), np.uint8)
    for poly in row["segmentation"]:
        pts = np.round(np.array(poly, np.float64).reshape(-1, 2)).astype(np.int32)
        cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def letterbox(img, size):
    """PIL RGB -> uint8 (size, size, 3), resized like SamResize (PIL bilinear), top-left, mean padding."""
    w, h = img.size
    s = size / max(w, h)
    nw, nh = int(w * s + 0.5), int(h * s + 0.5)
    canvas = Image.new("RGB", (size, size), MEAN_RGB)
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (0, 0))
    return np.asarray(canvas)


# ---------------------------------------------------------------- encoders

class TorchEncoder:
    def __init__(self, sam, student=None, device="mps"):
        from convert import EncoderWrapper
        self.dev = device
        enc = torch.load(student, weights_only=False) if student else EncoderWrapper(sam)
        self.m = enc.eval().to(device)

    @torch.no_grad()
    def __call__(self, x):
        return self.m(torch.from_numpy(x).to(self.dev)).float().cpu().numpy()


class CoreMLEncoder:
    def __init__(self, path, unit):
        import coremltools as ct
        units = {"cpu": ct.ComputeUnit.CPU_ONLY, "gpu": ct.ComputeUnit.CPU_AND_GPU,
                 "ane": ct.ComputeUnit.CPU_AND_NE, "all": ct.ComputeUnit.ALL}
        self.m = ct.models.MLModel(path, compute_units=units[unit])
        spec = self.m.get_spec().description.input[0]
        self.name, self.image_input = spec.name, spec.type.WhichOneof("Type") == "imageType"

    def __call__(self, x):
        inp = Image.fromarray(x[0].transpose(1, 2, 0).astype(np.uint8)) if self.image_input else x
        out = self.m.predict({self.name: inp})
        return np.asarray(list(out.values())[0], np.float32)


class CoreAIEncoder:
    def __init__(self, path, unit):
        import coreai.runtime as rt
        self.rt = rt
        kind = {"cpu": rt.ComputeUnitKind.cpu, "gpu": rt.ComputeUnitKind.gpu, "ane": rt.ComputeUnitKind.neural_engine}[unit]()
        opts = rt.SpecializationOptions.from_preferred_compute_unit_kind(kind)
        self.loop = asyncio.new_event_loop()
        model = self.loop.run_until_complete(rt.AIModel.load(path, opts))
        self.f = model.load_function(model.function_names[0])
        self.name = self.f.desc.input_names[0]
        self.dtype = np.float16 if "float16" in str(self.f.desc.input_descriptor(self.name)).lower() else np.float32

    def __call__(self, x):
        out = self.loop.run_until_complete(self.f({self.name: self.rt.NDArray(np.ascontiguousarray(x, self.dtype))}))
        return np.asarray(list(out.values())[0].numpy(), np.float32)


class CoreAISplitEncoder:
    """Two Core AI models (tools/split_encoder.py): A's outputs feed B's inputs in order. unit = "ane,gpu" etc."""

    def __init__(self, paths, units):
        pa, pb = paths.split(",")
        ua, ub = (units.split(",") + [units])[:2] if "," in units else (units, units)
        self.a, self.b = CoreAIEncoder(pa, ua), CoreAIEncoder(pb, ub)

    def __call__(self, x):
        a = self.a
        out = a.loop.run_until_complete(a.f({a.name: a.rt.NDArray(np.ascontiguousarray(x, a.dtype))}))
        feats = [out[n] for n in a.f.desc.output_names]
        b = self.b
        res = b.loop.run_until_complete(b.f(dict(zip(b.f.desc.input_names, feats))))
        return np.asarray(list(res.values())[0].numpy(), np.float32)


# ---------------------------------------------------------------- scoring

@torch.no_grad()
def decode(dec, sam, emb, row, size):
    s = 1024 / max(row["w"], row["h"])
    x, y, bw, bh = row["bbox"]
    box = torch.tensor([[[x * s, y * s], [(x + bw) * s, (y + bh) * s]]], dtype=torch.float32)
    low, _ = dec(torch.from_numpy(emb), box, torch.zeros(1, 1, 256, 256), torch.zeros(1))
    m = F.interpolate(low, (1024, 1024), mode="bilinear", align_corners=False)
    ih, iw = int(row["h"] * s + 0.5), int(row["w"] * s + 0.5)
    m = F.interpolate(m[..., :ih, :iw], (row["h"], row["w"]), mode="bilinear", align_corners=False)
    return (m[0, 0] > 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["torch", "coreml", "coreai", "coreai-split"], default="torch")
    ap.add_argument("--encoder", help="mlpackage / aimodel path")
    ap.add_argument("--student", help="torch: path to a pickled encoder nn.Module (pixels in, embedding out)")
    ap.add_argument("--unit", default="gpu", help="cpu|gpu|ane|all; coreai-split: '<unit A>,<unit B>'")
    ap.add_argument("--model", default="l0")
    ap.add_argument("--split", default="eval", choices=["eval", "dev"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    from convert import DecoderWrapper, EncoderWrapper
    sam = load_sam(args.model)
    size = INPUT_SIZE[args.model]
    dec = DecoderWrapper(sam).eval()
    ref_enc = EncoderWrapper(sam).eval().to("mps")
    if args.backend == "torch":
        enc = TorchEncoder(sam, args.student)
    elif args.backend == "coreml":
        enc = CoreMLEncoder(args.encoder, args.unit)
    elif args.backend == "coreai":
        enc = CoreAIEncoder(args.encoder, args.unit)
    else:
        enc = CoreAISplitEncoder(args.encoder, args.unit)

    rows = split(args.split)[: args.n]
    ious, ref_ious, coss, times = [], [], [], []
    for i, row in enumerate(rows):
        img = Image.open(os.path.join(COCO, "val2017", row["file"])).convert("RGB")
        x = letterbox(img, size).transpose(2, 0, 1)[None].astype(np.float32)
        t = time.perf_counter()
        emb = enc(x).reshape(1, 256, 64, 64)
        times.append(time.perf_counter() - t)
        with torch.no_grad():
            ref = ref_enc(torch.from_numpy(x).to("mps")).cpu().numpy()
        gt = gt_mask(row)
        pred, pref = decode(dec, sam, emb, row, size), decode(dec, sam, ref, row, size)
        ious.append((pred & gt).sum() / max(1, (pred | gt).sum()))
        ref_ious.append((pref & gt).sum() / max(1, (pref | gt).sum()))
        e, r = emb.ravel().astype(np.float64), ref.ravel().astype(np.float64)
        coss.append(float(e @ r / (np.linalg.norm(e) * np.linalg.norm(r) + 1e-12)))
        if np.isnan(emb).any():
            coss[-1] = float("nan")
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)} mIoU={100 * np.mean(ious):.2f} ref={100 * np.mean(ref_ious):.2f} cos={np.nanmean(coss):.4f}", flush=True)

    res = dict(tag=args.tag, backend=args.backend, unit=args.unit, encoder=args.encoder or args.student, split=args.split,
               n=len(rows), mIoU=100 * float(np.mean(ious)), ref_mIoU=100 * float(np.mean(ref_ious)),
               delta=100 * float(np.mean(ious) - np.mean(ref_ious)), cos_mean=float(np.nanmean(coss)),
               cos_min=float(np.nanmin(coss)), nan_frac=float(np.mean(np.isnan(coss))),
               enc_ms_median=1000 * float(np.median(times[5:])) if len(times) > 5 else None)
    os.makedirs(os.path.join(RESULTS, "eval"), exist_ok=True)
    json.dump(res | dict(per_image_iou=[float(v) for v in ious], per_image_cos=coss),
              open(os.path.join(RESULTS, "eval", f"{args.tag}.json"), "w"))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
