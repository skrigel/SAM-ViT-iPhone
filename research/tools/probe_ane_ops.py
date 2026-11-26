"""Single-op numerics probes: what does each compute unit actually compute in fp16?

P1  activation error vs exact fp32 over x in [-8, 8]: gelu(tanh), gelu(exact), relu, sigmoid, tanh, x*sigmoid(1.702x)
P2  1x1 conv accumulation: output error when the true result is a small number obtained by cancelling large terms
    (as in the folded high-gain BatchNorm of stages.3.op_list.1.point_conv).

Usage: python tools/probe_ane_ops.py  -> results/probe_ane_ops.json
"""

import json
import os
import sys

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evitsam import RESULTS, ROOT  # noqa: E402

UNITS = {"cpu": ct.ComputeUnit.CPU_ONLY, "gpu": ct.ComputeUnit.CPU_AND_GPU, "ane": ct.ComputeUnit.CPU_AND_NE}
TMP = os.path.join(ROOT, "models", "probes")


def convert(module, shape, name):
    path = os.path.join(TMP, f"{name}.mlpackage")
    if not os.path.exists(path):
        traced = torch.jit.trace(module.eval(), torch.randn(*shape))
        ml = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=shape)], outputs=[ct.TensorType(name="y")],
                        compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.iOS18)
        ml.save(path)
    return path


class Act(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f
        # A 1x1 conv in front so the graph is big enough to be scheduled on the ANE (identity weights).
        self.c = nn.Conv2d(64, 64, 1, bias=False)
        with torch.no_grad():
            self.c.weight.copy_(torch.eye(64).view(64, 64, 1, 1))

    def forward(self, x):
        return self.f(self.c(x))


ACTS = {
    "gelu_tanh": lambda x: nn.functional.gelu(x, approximate="tanh"),
    "gelu_erf": lambda x: nn.functional.gelu(x),
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "silu1702": lambda x: x * torch.sigmoid(1.702 * x),
    "hardswish": nn.functional.hardswish,
}


def p1():
    os.makedirs(TMP, exist_ok=True)
    n = 64 * 128 * 128
    xs = np.linspace(-8, 8, n, dtype=np.float32)
    x16 = xs.astype(np.float16).astype(np.float32)  # what the model sees after fp16 input cast
    x = x16.reshape(1, 64, 128, 128)
    out = {}
    for name, f in ACTS.items():
        path = convert(Act(f), (1, 64, 128, 128), f"act_{name}")
        ref = f(torch.from_numpy(x).double()).numpy().ravel()
        row = {}
        for u, cu in UNITS.items():
            y = ct.models.MLModel(path, compute_units=cu).predict({"x": x})["y"].astype(np.float64).ravel()
            err = np.abs(y - ref)
            # fp16 output rounding alone gives err <= ulp(|ref|)/2
            ulp = np.spacing(np.abs(ref).astype(np.float16)).astype(np.float64)
            near0 = np.abs(x16) < 1
            row[u] = dict(max_abs=float(err.max()), max_abs_near0=float(err[near0].max()),
                          mean_abs=float(err.mean()), max_in_ulps=float((err / np.maximum(ulp, 1e-8)).max()),
                          argmax_x=float(x16[err.argmax()]))
        out[name] = row
        print(name, {u: f"max={r['max_abs']:.2e} (|x|<1: {r['max_abs_near0']:.2e}) ulps={r['max_in_ulps']:.1f}" for u, r in row.items()}, flush=True)
    return out


class Conv(nn.Module):
    def __init__(self, w, b):
        super().__init__()
        self.c = nn.Conv2d(w.shape[1], w.shape[0], 1)
        with torch.no_grad():
            self.c.weight.copy_(torch.from_numpy(w).view(*w.shape, 1, 1))
            self.c.bias.copy_(torch.from_numpy(b))

    def forward(self, x):
        return self.c(x)


def p2():
    """y = W x + b where W x ~ -K and b ~ +K, so y is O(1). Error vs K tells whether accumulation is fp16 or fp32."""
    rng = np.random.default_rng(0)
    cin, cout = 1024, 64
    x = (1.8 + 0.005 * rng.standard_normal((1, cin, 32, 32))).astype(np.float16).astype(np.float32)
    out = {}
    for k in [1, 16, 64, 256, 584]:
        w = (rng.standard_normal((cout, cin)) * k / (1.8 * np.sqrt(cin))).astype(np.float16).astype(np.float32)
        b = (-(w @ np.full(cin, 1.8, np.float32)) + rng.standard_normal(cout)).astype(np.float32)
        path = convert(Conv(w, b), x.shape, f"conv_cancel_{k}")
        ref = (np.einsum("oc,bchw->bohw", w.astype(np.float64), x.astype(np.float64)) + b[None, :, None, None])
        mag = float(np.abs(np.einsum("oc,bchw->bohw", w.astype(np.float64), x.astype(np.float64))).mean())
        row = {"|Wx|": mag, "std(y)": float(ref.std())}
        for u, cu in UNITS.items():
            y = ct.models.MLModel(path, compute_units=cu).predict({"x": x})["y"].astype(np.float64)
            e = y - ref
            row[u] = dict(rmse=float(np.sqrt((e ** 2).mean())), max_abs=float(np.abs(e).max()),
                          cos=float((y.ravel() - y.mean()) @ (ref.ravel() - ref.mean()) / (np.linalg.norm(y - y.mean()) * np.linalg.norm(ref - ref.mean()))))
        out[f"K={k}"] = row
        print(f"K={k} |Wx|={mag:.1f}", {u: f"rmse={row[u]['rmse']:.3g} cos={row[u]['cos']:.4f}" for u in UNITS}, flush=True)
    return out


if __name__ == "__main__":
    res = {"activations": p1(), "conv_cancellation": p2()}
    json.dump(res, open(os.path.join(RESULTS, "probe_ane_ops.json"), "w"), indent=1)
