"""Fixed-shape decoder (convert.DecoderWrapper, box prompt) vs the app's flexible-shape decoder: placement, latency,
and agreement with the PyTorch fp32 decoder on dev images. Python-side timing (relative only).

  python tools/decoder_placement.py [--n 100]  -> results/decoder_placement.json
"""
import argparse, collections, json, os, sys, time
import coremltools as ct
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert import DecoderWrapper, EncoderWrapper  # noqa: E402
from eval_coco import COCO, letterbox, split  # noqa: E402
from eval_decoder import APP_DECODER  # noqa: E402
from evitsam import RESULTS, ROOT, load_sam  # noqa: E402
from coremltools.models.compute_plan import MLComputePlan  # noqa: E402

UNITS = {"cpu": ct.ComputeUnit.CPU_ONLY, "gpu": ct.ComputeUnit.CPU_AND_GPU, "ane": ct.ComputeUnit.CPU_AND_NE, "all": ct.ComputeUnit.ALL}
FIXED = os.path.join(ROOT, "models", "EViTSAM_Decoder_l0_fp16.mlpackage")


def placement(path, unit):
    plan = MLComputePlan.load_from_path(path=ct.models.utils.compile_model(path), compute_units=UNITS[unit])
    cnt = collections.Counter()
    for op in plan.model_structure.program.functions["main"].block.operations:
        u = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if u:
            cnt[type(u.preferred_compute_device).__name__.replace("ML", "").replace("ComputeDevice", "")] += 1
    return dict(cnt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    sam = load_sam("l0")
    enc, dec = EncoderWrapper(sam).eval().to("mps"), DecoderWrapper(sam).eval()
    cases = []
    for row in split("dev")[: a.n]:
        x = letterbox(Image.open(os.path.join(COCO, "val2017", row["file"])).convert("RGB"), 512).transpose(2, 0, 1)[None].astype(np.float32)
        with torch.no_grad():
            emb = enc(torch.from_numpy(x).to("mps")).cpu()
        s = 1024 / max(row["w"], row["h"])
        bx, by, bw, bh = row["bbox"]
        box = np.array([[[bx * s, by * s], [(bx + bw) * s, (by + bh) * s]]], np.float32)
        with torch.no_grad():
            low, _ = dec(emb, torch.from_numpy(box), torch.zeros(1, 1, 256, 256), torch.zeros(1))
        cases.append((emb.numpy(), box, low[0, 0].numpy() > 0))
    out = {}
    for name, path in [("fixed", FIXED), ("app_flexible", APP_DECODER)]:
        for u in ["cpu", "gpu", "ane", "all"]:
            m = ct.models.MLModel(path, compute_units=UNITS[u])
            ious, ts = [], []
            for emb, box, ref in cases:
                if name == "fixed":
                    feed = {"image_embeddings": emb, "box_coords": box, "mask_input": np.zeros((1, 1, 256, 256), np.float32),
                            "has_mask_input": np.zeros(1, np.float32)}
                else:
                    feed = {"image_embeddings": emb.astype(np.float16), "point_coords": box.astype(np.float16),
                            "point_labels": np.array([[2, 3]], np.float16)}
                t = time.perf_counter()
                o = m.predict(feed)
                ts.append(1000 * (time.perf_counter() - t))
                mk = (o["low_res_masks"][0, 0] if name == "fixed" else o["masks"][0, 0]) > 0
                ious.append((mk & ref).sum() / max(1, (mk | ref).sum()))
            r = dict(placement=placement(path, u), ms_median=float(np.median(ts[5:])), agree_iou=float(np.mean(ious)),
                     agree_min=float(np.min(ious)))
            out[f"{name}/{u}"] = r
            print(name, u, json.dumps(r), flush=True)
    json.dump(out, open(os.path.join(RESULTS, "decoder_placement.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
