"""Shared helpers: import EfficientViT on macOS (no triton) and load checkpoints."""

import os
import sys
import types

RESEARCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Large, untracked artifacts (checkpoints, data, converted models) live under work/ (see init.sh).
ROOT = os.environ.get("SAM_WORK", os.path.join(RESEARCH, "work"))
RESULTS = os.path.join(RESEARCH, "results")
sys.path.insert(0, os.path.join(ROOT, "third_party", "efficientvit"))

import torch._dynamo  # noqa: E402,F401  (before the stub below, so torch's own Triton detection sees no Triton)

# EfficientViT imports a Triton RMSNorm kernel at module load; Triton is CUDA-only and unused by SAM.
if "triton" not in sys.modules:
    _triton = types.ModuleType("triton")
    _triton.jit = lambda fn=None, **_: fn if fn is not None else (lambda f: f)
    _triton.cdiv = lambda a, b: (a + b - 1) // b
    _tl = types.ModuleType("triton.language")
    _tl.constexpr = object
    _triton.language = _tl
    sys.modules["triton"] = _triton
    sys.modules["triton.language"] = _tl

from efficientvit.sam_model_zoo import create_efficientvit_sam_model  # noqa: E402

# Drop the stub again so torch (export / inductor) does not mistake it for a real Triton install.
if getattr(sys.modules.get("triton"), "__file__", None) is None:
    sys.modules.pop("triton", None)
    sys.modules.pop("triton.language", None)

INPUT_SIZE = {"l0": 512, "l1": 512, "l2": 512, "xl0": 1024, "xl1": 1024}


def load_sam(name):
    ckpt = os.path.join(ROOT, "checkpoints", f"efficientvit_sam_{name}.pt")
    return create_efficientvit_sam_model(f"efficientvit-sam-{name}", pretrained=True, weight_url=ckpt).eval()


import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from efficientvit.models.nn.ops import UpSampleLayer  # noqa: E402


class SeparableResize(nn.Module):
    """Exact replacement for a fixed-size bicubic/bilinear F.interpolate as two matmuls.

    Core ML has no bicubic upsample op; interpolation along each axis is linear, so
    out = A_h @ x @ A_w^T with A built by interpolating an identity matrix.
    """

    def __init__(self, size, mode, align_corners):
        super().__init__()
        self.size, self.mode, self.align_corners = tuple(size), mode, align_corners
        self._cache = {}

    def _axis_matrix(self, n, m):
        # Cached so that, after one warm-up call, tracing sees a constant instead of an interpolate op.
        if (n, m) not in self._cache:
            eye = torch.eye(n).view(1, 1, n, n)
            a = F.interpolate(eye, size=(m, n), mode=self.mode, align_corners=self.align_corners)[0, 0]
            self._cache[(n, m)] = a.detach().clone()
        return self._cache[(n, m)]

    def forward(self, x):
        h, w = int(x.shape[-2]), int(x.shape[-1])
        if (h, w) == self.size:
            return x
        a_h = self._axis_matrix(h, self.size[0]).to(x.device, x.dtype)
        a_w = self._axis_matrix(w, self.size[1]).to(x.device, x.dtype)
        # Two `linear` ops on the last axis, not matmul(a_h, x) @ a_w.T: the Neural Engine compiler miscompiles the
        # matmul form when its result feeds the neck's add (embedding cosine 0.19; 0.998 as linear). Same math.
        y = F.linear(x, a_w)
        return F.linear(y.transpose(-1, -2), a_h).transpose(-1, -2)


from efficientvit.models.nn.ops import LiteMLA  # noqa: E402


def fp16_safe_relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
    """LiteMLA's ReLU linear attention rearranged so it is correct in fp16 on CPU, GPU and Neural Engine.

    out_t = (V K^T q_t) / (1^T K^T q_t) is unchanged by scaling K by any per-head constant and each q_t by a
    positive scalar. So K is normalized per head and averaged over tokens, each q_t is normalized to max 1, and
    the denominator is scaled by FP16_PAD (then divided back out). The reference upcasts this block to fp32;
    in pure fp16, V K^T overflows (1.3e6 in L2) and the denominator goes subnormal (5e-5 in L0), which the
    Neural Engine flushes to zero. Its eps of 1e-15 also rounds to 0, so eps here is fp16-representable.

    Layout matters on the Neural Engine: the reference's strided q/k/v slices of a 4-D reshape followed by
    F.pad miscompile there (NaN output), so q/k/v come from a 5-D reshape and the denominator is computed
    separately instead of as a padded row of V.
    """
    B, _, H, W = list(qkv.size())
    n = int(H) * int(W)  # static shapes: a constant
    qkv = torch.reshape(qkv, (B, -1, 3, self.dim, n))
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    q = self.kernel_func(q)
    k = self.kernel_func(k)
    k = k / (torch.amax(k, dim=(2, 3), keepdim=True) + 1e-4) * (1.0 / n)
    q = q / (torch.amax(q, dim=2, keepdim=True) + 1e-4)
    num = torch.matmul(torch.matmul(v, k.transpose(-1, -2)), q)
    den = torch.matmul(torch.sum(k, dim=3, keepdim=True).transpose(-1, -2), q) * FP16_PAD
    out = num / (den + 1e-4) * FP16_PAD
    return torch.reshape(out, (B, -1, H, W))


FP16_PAD = 1024.0


def make_coreml_friendly(module):
    for name, child in module.named_children():
        if isinstance(child, UpSampleLayer) and child.size is not None:
            setattr(module, name, SeparableResize(child.size, child.mode, child.align_corners))
        else:
            if isinstance(child, LiteMLA):
                child.relu_linear_att = types.MethodType(fp16_safe_relu_linear_att, child)
            make_coreml_friendly(child)
    return module


# ---------------------------------------------------------------------------------------------- ANE-accurate GELU
#
# The Neural Engine's built-in GELU (both the erf and tanh forms lower to it) is off by up to 6.2e-3, largest near 0
# (results/probe_ane_ops.json), and the high-gain folded BatchNorms in stages 3-4 amplify that error up to 321x.
# Its ReLU, add and mul are exact up to fp16 rounding. So GELU is rebuilt from those:
#   gelu(x) = relu(x) - g(|x|),   g(t) = t - gelu(t) = -gelu(-t)   (exact identity, also for the tanh form)
# g is a smooth bump (max 0.17 at t~0.75, g(4) = 7e-5), so it is fitted by a Chebyshev series on [0, T] with |x| clamped
# to T, and evaluated with Clenshaw's recurrence, whose intermediates stay small enough to be accurate in fp16.


def _gelu_tanh_np(x):
    import numpy as np
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def _gelu_erf_np(x):
    import math
    import numpy as np
    return 0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))


class ClenshawGELU(nn.Module):
    def __init__(self, approximate="tanh", degree=8, T=4.0):
        super().__init__()
        import numpy as np
        t = np.linspace(0, T, 20001)
        g = t - (_gelu_tanh_np(t) if approximate == "tanh" else _gelu_erf_np(t))
        c = np.polynomial.chebyshev.chebfit(2 * t / T - 1, g, degree)
        self.coef = [float(v) for v in c]
        self.T = float(T)

    def forward(self, x):
        u = torch.clamp(torch.abs(x), max=self.T) * (2.0 / self.T) - 1.0
        u2 = u * 2.0
        b1 = torch.zeros_like(u)
        b2 = torch.zeros_like(u)
        for ck in self.coef[:0:-1]:
            b0 = u2 * b1 - b2 + ck
            b2, b1 = b1, b0
        g = u * b1 - b2 + self.coef[0]
        return torch.relu(x) - g


def replace_gelu(module, factory, where=lambda name: True, prefix=""):
    """Swap every nn.GELU whose qualified name passes `where` for factory(gelu_module). Returns the names replaced."""
    done = []
    for name, child in module.named_children():
        q = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.GELU) and where(q):
            setattr(module, name, factory(child))
            done.append(q)
        else:
            done += replace_gelu(child, factory, where, q)
    return done


class TanhFormGELU(nn.Module):
    """gelu_tanh written out as 0.5x + 0.5x * tanh(c x + c k x^3): same function, ~8 elementwise ops.

    Written this way, coremltools does not fuse it back into the `gelu` op, so the Neural Engine runs its tanh
    (error ~1.6e-3) instead of its GELU (6.2e-3). The tanh error is multiplied by 0.5|x|, so it vanishes near 0
    where most activations are (3e-5 for |x| < 0.1 vs 3e-3 for the built-in GELU). Cheaper than ClenshawGELU,
    slightly less accurate for |x| > 1.
    """

    C = 0.7978845608028654  # sqrt(2/pi)

    def forward(self, x):
        h = 0.5 * x
        return h + h * torch.tanh(self.C * x + (self.C * 0.044715) * (x * x * x))
