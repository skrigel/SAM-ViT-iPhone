"""Static compute profile (FLOPs, params, memory traffic) per stage and per op class, from PyTorch.

Usage: python tools/profile_static.py  -> results/static.json
"""

import json
import os
import sys
from collections import defaultdict

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert import DecoderWrapper, EncoderWrapper  # noqa: E402
from evitsam import INPUT_SIZE, RESULTS, ROOT, load_sam, make_coreml_friendly  # noqa: E402

BYTES = 2  # fp16 on device


def conv_class(m: nn.Conv2d) -> str:
    k = m.kernel_size[0]
    if m.groups == m.in_channels and m.groups > 1:
        return f"depthwise conv {k}x{k}"
    if m.groups > 1:
        return f"grouped conv {k}x{k}"
    return "pointwise conv 1x1" if k == 1 else f"dense conv {k}x{k}"


def stage_of(name: str) -> str:
    parts = name.split(".")
    # encoder.backbone.stages.3.op_list.1.context_module... -> backbone.stage3
    if "backbone" in parts:
        i = parts.index("backbone")
        if parts[i + 1] == "input_stem":
            return "backbone.stem"
        if parts[i + 1] == "stages":
            return f"backbone.stage{int(parts[i + 2]) + 1}"
    if "neck" in parts:
        i = parts.index("neck")
        return {"inputs": "neck.fuse(1x1+upsample)", "middle": "neck.fmbconv blocks", "outputs": "neck.out"}.get(
            parts[i + 1], "neck"
        )
    if parts[-1] == "norm" and "encoder" in parts[:2]:
        return "out layernorm"
    return parts[1] if len(parts) > 1 else name


def block_kind(name: str) -> str:
    if "context_module" in name:
        return "attention (LiteMLA)"
    if "local_module" in name:
        return "MBConv (in attention block)"
    return "conv blocks"


def profile(model, inputs, stage_fn):
    records = []
    hooks = []

    def hook(name):
        def fn(m, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            in_b = x.numel() * BYTES if torch.is_tensor(x) else 0
            outs = out if isinstance(out, (tuple, list)) else (out,)
            out_b = sum(o.numel() for o in outs if torch.is_tensor(o)) * BYTES
            w_b = sum(p.numel() for p in m.parameters(recurse=False)) * BYTES
            flops, cls = 0, type(m).__name__
            o = outs[0]
            if isinstance(m, nn.Conv2d):
                cls = conv_class(m)
                flops = 2 * o.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
            elif isinstance(m, nn.Linear):
                cls = "linear / matmul"
                flops = 2 * o.numel() * m.in_features
            elif isinstance(m, (nn.BatchNorm2d,)):
                cls = "batchnorm (fused at convert)"
            elif isinstance(m, (nn.LayerNorm,)) or "LayerNorm" in type(m).__name__:
                cls = "layernorm"
                flops = 5 * o.numel()
            elif isinstance(m, (nn.GELU, nn.ReLU, nn.Hardswish, nn.SiLU, nn.ReLU6)):
                cls = f"activation ({type(m).__name__})"
                flops = o.numel()
            records.append(
                dict(name=name, cls=cls, stage=stage_fn(name), kind=block_kind(name), flops=flops,
                     params=w_b // BYTES, bytes=in_b + out_b + w_b, out_shape=list(o.shape))
            )
        return fn

    for name, m in model.named_modules():
        if len(list(m.children())) == 0:
            hooks.append(m.register_forward_hook(hook(name)))
    # Attention matmuls / resize matmuls are functional (torch.matmul), count them via FlopCounterMode per module.
    from torch.utils.flop_counter import FlopCounterMode

    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(*inputs)
    for h in hooks:
        h.remove()
    counted = defaultdict(int)
    for r in records:
        counted[r["name"]] += r["flops"]
    # FlopCounterMode reports per module hierarchy; extract matmul FLOPs attributed to non-leaf modules.
    per_mod = fc.get_flop_counts()
    extra = []
    for mod_name, ops in per_mod.items():
        mm = sum(v for k, v in ops.items() if "mm" in str(k) or "matmul" in str(k) or "bmm" in str(k))
        if mm == 0:
            continue
        # keep only the deepest module that owns the matmul: skip if any child also reports matmul flops
        deeper = [n for n in per_mod if n != mod_name and n.startswith(mod_name + ".")
                  and sum(v for k, v in per_mod[n].items() if "mm" in str(k))]
        if deeper:
            continue
        name = mod_name.replace("Global.", "").replace("Global", "")
        name = name.split(".", 1)[1] if name.startswith(type(model).__name__) else name
        linear_in_mod = sum(r["flops"] for r in records if r["name"].startswith(name) and r["cls"] == "linear / matmul")
        mm -= linear_in_mod
        if mm > 0:
            cls = "attention matmul" if ("context_module" in name or "attn" in name or "transformer" in name) else "resize matmul (bicubic)"
            extra.append(dict(name=name + " (functional matmul)", cls=cls, stage=stage_fn(name), kind=block_kind(name),
                              flops=mm, params=0, bytes=0, out_shape=[]))
    return records + extra, fc.get_total_flops()


def summarize(records):
    by = lambda key: {k: dict(flops=v[0], params=v[1], bytes=v[2], n=v[3]) for k, v in _agg(records, key).items()}
    top = sorted([r for r in records if r["flops"] > 0], key=lambda r: -r["flops"])[:25]
    return dict(
        total_flops=sum(r["flops"] for r in records),
        total_params=sum(r["params"] for r in records),
        total_bytes=sum(r["bytes"] for r in records),
        by_stage=by("stage"),
        by_class=by("cls"),
        by_kind=by("kind"),
        top_layers=[dict(name=r["name"], cls=r["cls"], flops=r["flops"], bytes=r["bytes"],
                         intensity=(r["flops"] / r["bytes"]) if r["bytes"] else None, out_shape=r["out_shape"])
                    for r in top],
    )


def _agg(records, key):
    out = defaultdict(lambda: [0, 0, 0, 0])
    for r in records:
        a = out[r[key]]
        a[0] += r["flops"]; a[1] += r["params"]; a[2] += r["bytes"]; a[3] += 1
    return dict(out)


def main():
    results = {}
    for name in ["l0", "l2", "xl0"]:
        sam = load_sam(name)
        size = INPUT_SIZE[name]
        enc = make_coreml_friendly(EncoderWrapper(sam)).eval()
        recs, total = profile(enc, (torch.rand(1, 3, size, size) * 255,), lambda n: stage_of("encoder." + n.split(".", 1)[1]) if n.startswith("encoder") else n)
        results[f"encoder_{name}"] = dict(input=size, flopcounter_total=total, **summarize(recs))
        print(name, "encoder GFLOPs", round(results[f"encoder_{name}"]["total_flops"] / 1e9, 1), "fc", round(total / 1e9, 1))
        if name == "l0":
            dec = DecoderWrapper(sam).eval()
            ex = (torch.randn(1, 256, 64, 64), torch.tensor([[[100.0, 100.0], [600.0, 500.0]]]),
                  torch.zeros(1, 1, 256, 256), torch.ones(1))

            def dstage(n):
                p = n.split(".")
                if p[0] == "pe":
                    return "prompt encoder (mask downscaling)"
                if "transformer" in p:
                    return "two-way transformer"
                if "output_upscaling" in p:
                    return "mask upscaling (transposed conv)"
                if "output_hypernetworks_mlps" in p:
                    return "hypernetwork MLPs"
                if "iou_prediction_head" in p:
                    return "IoU head"
                return p[1] if len(p) > 1 else n

            recs, total = profile(dec, ex, dstage)
            results["decoder"] = dict(flopcounter_total=total, **summarize(recs))
            print("decoder GFLOPs", round(results["decoder"]["total_flops"] / 1e9, 2), "fc", round(total / 1e9, 2))

    os.makedirs(os.path.join(RESULTS), exist_ok=True)
    with open(os.path.join(RESULTS, "static.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
