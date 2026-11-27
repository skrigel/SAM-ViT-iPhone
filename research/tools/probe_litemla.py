"""Where inside LiteMLA does the ANE go wrong? Taps every intermediate of one LiteMLA block on chosen dev images.

Captures the block's fp32 input from the PyTorch encoder (or a pickled student), converts the block alone to Core ML fp16
with every intermediate as an output, and compares GPU/ANE against fp32 per intermediate. Also prints magnitudes.

  python tools/probe_litemla.py --student work/students/relu_recal_awake_120m.pt --rows 17,78,0
"""
import argparse, json, os, sys
import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert import EncoderWrapper  # noqa: E402
from eval_coco import COCO, letterbox, split  # noqa: E402
from evitsam import FP16_PAD, RESULTS, ROOT, load_sam, make_coreml_friendly  # noqa: E402


class Probe(nn.Module):
    """LiteMLA.forward + fp16_safe_relu_linear_att, returning every intermediate."""

    def __init__(self, att, kscale=1.0):
        super().__init__()
        self.a, self.s = att, kscale

    def forward(self, x):
        a = self.a
        qkv = a.qkv(x)
        ms = [qkv] + [op(qkv) for op in a.aggreg]
        qkv = torch.cat(ms, dim=1)
        B, _, H, W = qkv.shape
        n = int(H) * int(W)
        r = torch.reshape(qkv, (B, -1, 3, a.dim, n))
        q, k, v = r[:, :, 0], r[:, :, 1], r[:, :, 2]
        q, k = a.kernel_func(q), a.kernel_func(k)
        kmax = torch.amax(k, dim=(2, 3), keepdim=True)
        kn = k / (kmax + 1e-4) * (self.s / n)
        qmax = torch.amax(q, dim=2, keepdim=True)
        qn = q / (qmax + 1e-4)
        vk = torch.matmul(v, kn.transpose(-1, -2))
        num = torch.matmul(vk, qn)
        ksum = torch.sum(kn, dim=3, keepdim=True)
        pad = FP16_PAD / self.s
        den = torch.matmul(ksum.transpose(-1, -2), qn) * pad
        att = num / (den + 1e-4) * pad
        out = a.proj(torch.reshape(att, (B, -1, H, W)))
        return qkv, v, kmax, qmax, kn, qn, vk, num, ksum, den, att, out


NAMES = ["qkv_cat", "v", "kmax", "qmax", "k_norm", "q_norm", "vk", "num", "ksum", "den", "att", "out"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student")
    ap.add_argument("--block", default="encoder.backbone.stages.4.op_list.1.context_module")
    ap.add_argument("--rows", default="17,78,0")
    ap.add_argument("--tag", default="litemla_probe")
    ap.add_argument("--kscale", type=float, default=1.0, help="scale K by S and the den pad by 1/S (exact rewrite)")
    args = ap.parse_args()
    enc = (torch.load(args.student, weights_only=False) if args.student else EncoderWrapper(load_sam("l0"))).eval()
    mods = dict(enc.named_modules())
    blk = mods[args.block]
    caps = []
    h = blk.register_forward_hook(lambda m, i, o: caps.append(i[0].detach().clone()))
    dev = split("dev")
    rows = [int(r) for r in args.rows.split(",")]
    for r in rows:
        img = Image.open(os.path.join(COCO, "val2017", dev[r]["file"])).convert("RGB")
        with torch.no_grad():
            enc(torch.from_numpy(letterbox(img, 512).transpose(2, 0, 1)[None].astype(np.float32)))
    h.remove()
    probe = Probe(blk.main if hasattr(blk, "main") else blk, args.kscale).eval()  # context_module = ResidualBlock(LiteMLA)
    traced = torch.jit.trace(probe, caps[0])
    path = os.path.join(ROOT, "models", "taps", f"{args.tag}.mlpackage")
    ml = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=caps[0].shape)],
                    outputs=[ct.TensorType(name=n) for n in NAMES],
                    compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.iOS18)
    ml.save(path)
    res = {}
    for j, (r, x) in enumerate(zip(rows, caps)):
        with torch.no_grad():
            ref = [t.numpy().astype(np.float64) for t in probe(x)]
        print(f"\n== dev row {r}: input max|x| {x.abs().max():.1f}")
        print(f"{'tensor':10s} {'ref max|.|':>11s} {'ref min|.|>0':>12s}" + "".join(f"{u+' cos':>10s}{u+' relerr':>11s}" for u in ("gpu", "ane")))
        outs = {}
        for u, cu in (("gpu", ct.ComputeUnit.CPU_AND_GPU), ("ane", ct.ComputeUnit.CPU_AND_NE)):
            m = ct.models.MLModel(path, compute_units=cu)
            outs[u] = m.predict({"x": x.numpy()})
        rows_out = []
        for i, n in enumerate(NAMES):
            a = ref[i].ravel()
            nz = np.abs(a[a != 0])
            line = f"{n:10s} {np.abs(a).max():11.3e} {(nz.min() if nz.size else 0):12.3e}"
            d = {"ref_max": float(np.abs(a).max())}
            for u in ("gpu", "ane"):
                b = np.asarray(outs[u][n], np.float64).ravel()
                cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
                rel = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30))
                line += f"{cos:10.4f}{rel:11.2e}"
                d[u] = dict(cos=cos, rel=rel)
            print(line)
            rows_out.append(d)
        res[r] = dict(zip(NAMES, rows_out))
    os.makedirs(os.path.join(RESULTS, "taps"), exist_ok=True)
    json.dump(res, open(os.path.join(RESULTS, "taps", f"{args.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
