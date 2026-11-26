"""Distill an ANE-friendly student encoder from the stock EfficientViT-SAM encoder (proposal section 4.1).

Student = teacher with every GELU swapped for an activation the Neural Engine computes exactly (ReLU by default),
initialised from the teacher's weights and trained to match the teacher's image embedding with an L2 loss
(the objective EfficientViT-SAM uses in its first training stage). Data: COCO train2017 images (never val2017).

  python tools/train_distill.py --act relu --minutes 120 --tag relu_v1     # fixed wall-clock budget (PROGRAM.md)
  -> work/students/<tag>.pt (whole EncoderWrapper module, pixels in), work/logs/distill_<tag>.log

Evaluate:  python tools/eval_coco.py --backend torch --student work/students/<tag>.pt --tag <tag>_torch
Convert:   python tools/convert.py --student work/students/<tag>.pt --precision fp16 --encoder-only --name <tag>_fp16
"""

import argparse
import copy
import glob
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert import EncoderWrapper  # noqa: E402
from eval_coco import MEAN_RGB  # noqa: E402
from evitsam import INPUT_SIZE, ROOT, load_sam, replace_gelu  # noqa: E402

ACTS = {"relu": nn.ReLU, "relu6": lambda: nn.ReLU6(), "hardswish": nn.Hardswish}


class Images(torch.utils.data.Dataset):
    """Random-scale letterbox (top-left, mean padding, like inference) + horizontal flip."""

    def __init__(self, files, size):
        self.files, self.size = files, size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        w, h = img.size
        # Mostly the inference geometry (long side = size), sometimes a zoomed crop for variety.
        if random.random() < 0.3:
            s = random.uniform(1.0, 2.0) * self.size / max(w, h)
            nw, nh = max(1, int(w * s)), max(1, int(h * s))
            img = img.resize((nw, nh), Image.BILINEAR)
            x0, y0 = random.randint(0, max(0, nw - self.size)), random.randint(0, max(0, nh - self.size))
            img = img.crop((x0, y0, x0 + min(nw, self.size), y0 + min(nh, self.size)))
        else:
            s = self.size / max(w, h)
            img = img.resize((int(w * s + 0.5), int(h * s + 0.5)), Image.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), MEAN_RGB)
        canvas.paste(img, (0, 0))
        return torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="l0")
    ap.add_argument("--act", default="relu", choices=list(ACTS))
    ap.add_argument("--where", default="", help="regex on GELU names to swap (default: all)")
    ap.add_argument("--init", help="start from this student instead of the teacher")
    ap.add_argument("--minutes", type=float, default=120)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--bn", default="frozen", choices=["frozen", "train"],
                    help="frozen: BatchNorm uses (and keeps) running stats, as at inference; train: batch stats (bad at batch 8)")
    ap.add_argument("--recalib", type=int, default=200, help="batches used to re-estimate BN running stats before training")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    dev = "mps"
    size = INPUT_SIZE[args.model]
    teacher = EncoderWrapper(load_sam(args.model)).eval().to(dev)
    for p in teacher.parameters():
        p.requires_grad_(False)
    if args.init:
        student = torch.load(args.init, weights_only=False)
    else:
        import re
        student = copy.deepcopy(teacher)
        swapped = replace_gelu(student, lambda g: ACTS[args.act](), where=lambda n: re.search(args.where, n) is not None)
        print(f"swapped {len(swapped)} GELUs for {args.act}", flush=True)
    student = student.to(dev).train()
    bns = [m for m in student.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for p in student.parameters():
        p.requires_grad_(True)
    student.mean.requires_grad_(False)
    student.std.requires_grad_(False)

    files = sorted(glob.glob(os.path.join(ROOT, "data", "coco", "train2017", "*.jpg")))
    print(f"{len(files)} training images", flush=True)
    loader = torch.utils.data.DataLoader(Images(files, size), batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                         drop_last=True, persistent_workers=True, prefetch_factor=4)
    if args.recalib and not args.init:
        # The swapped activation changes every BN input distribution; re-estimate running stats (cumulative average).
        for m in bns:
            m.reset_running_stats()
            m.momentum = None
        with torch.no_grad():
            for i, x in enumerate(loader):
                if i >= args.recalib:
                    break
                student(x.to(dev))
        for m in bns:
            m.momentum = 0.1
        print(f"recalibrated BN stats on {args.recalib} batches", flush=True)
    if args.bn == "frozen":
        for m in bns:
            m.eval()
    with torch.no_grad():
        x = next(iter(loader)).to(dev)
        cos0 = F.cosine_similarity(student.eval()(x).flatten(1), teacher(x).flatten(1)).mean().item()
    student.train()
    if args.bn == "frozen":
        for m in bns:
            m.eval()
    print(f"inference-mode cosine before training: {cos0:.4f}", flush=True)
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd)

    out = os.path.join(ROOT, "students")
    os.makedirs(out, exist_ok=True)
    budget = args.minutes * 60
    t0 = time.time()
    step, ema = 0, None
    done = False
    while not done:
        for x in loader:
            frac = (time.time() - t0) / budget
            if frac >= 1:
                done = True
                break
            lr = args.lr * (min(1.0, step / 200) * 0.5 * (1 + math.cos(math.pi * frac)))  # warmup + cosine on wall clock
            for g in opt.param_groups:
                g["lr"] = lr
            x = x.to(dev, non_blocking=True)
            with torch.no_grad():
                target = teacher(x)
            pred = student(x)
            loss = F.mse_loss(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip:
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip)
            opt.step()
            step += 1
            ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
            if step % 50 == 0:
                with torch.no_grad():  # with --bn frozen this is the inference-mode cosine
                    cos = F.cosine_similarity(pred.flatten(1), target.flatten(1)).mean().item()
                print(f"step {step} t={time.time() - t0:.0f}s lr={lr:.2e} loss={ema:.5f} cos={cos:.5f}", flush=True)
            if step % 1000 == 0:
                torch.save(copy.deepcopy(student).cpu().eval(), os.path.join(out, f"{args.tag}.pt"))
    torch.save(copy.deepcopy(student).cpu().eval(), os.path.join(out, f"{args.tag}.pt"))
    print(f"done: {step} steps in {time.time() - t0:.0f}s, final loss {ema:.5f}", flush=True)
    os._exit(0)  # persistent DataLoader workers otherwise keep the process alive


if __name__ == "__main__":
    main()
