"""Models that bound what hand-fused kernels could ever save (timing only, outputs are not meaningful).

Fusion removes memory round trips and dispatches for the cheap ops (bias, BN, GELU, depthwise, residual); it cannot
remove the dense matmuls. So we time the stock-compiled graph with the cheap ops deleted:

  blk_<s3|s4>_dense    the MBConv block as x + W3'(W1 x + b1) + b3'  (BN folded, no GELU, no depthwise)
  enc_dense_mb         the full encoder with only the 8 fusable MBConv blocks made dense (bound for MBConv fusion)
  enc_dense_all        the full encoder with every activation and depthwise conv deleted and every conv+BN folded
                       (bound for fusing *everything* except dense convs/linears, attention matmuls and the resize)

stock - dense = the most a perfect fused kernel could save, assuming it runs the matmuls as well as the stock compiler.
Generous to fusion: the deleted ops are not free in any real kernel.

  peak_matmul_4096     (4096x4096) @ (4096x4096) fp16: the stock compiler's best-case matmul throughput
  peak_stream_64M      y = 1.5 x + 0.5 on 64M fp16 values (128 MB in, 128 MB out): device memory bandwidth

  USE_OS_COREAI=1 .venv-coreai/bin/python kernels/bound_models.py  -> work/models/<name>.aimodel
"""
import copy
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
from coreai_kernel_util import convert  # noqa: E402
from convert import EncoderWrapper  # noqa: E402
from evitsam import ROOT, load_sam, make_coreml_friendly  # noqa: E402

OUT = os.path.join(ROOT, "models")


def fold_bn(conv, bn):
    g = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    w = conv.weight * g.reshape(-1, 1, 1, 1)
    b = (conv.bias if conv.bias is not None else 0) * g + bn.bias - bn.running_mean * g
    new = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                    conv.dilation, conv.groups, bias=True)
    with torch.no_grad():
        new.weight.copy_(w)
        new.bias.copy_(b)
    return new


class DenseMBConv(nn.Module):
    def __init__(self, block):
        super().__init__()
        m = block.main
        self.inv = copy.deepcopy(m.inverted_conv.conv)
        self.proj = fold_bn(m.point_conv.conv, m.point_conv.norm)

    def forward(self, x):
        return x + self.proj(self.inv(x))


class Subsample(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.s = s

    def forward(self, x):
        return x[:, :, :: self.s, :: self.s]


def strip_all(mod):
    from efficientvit.models.nn.ops import ConvLayer
    for name, child in mod.named_children():
        if isinstance(child, (nn.GELU, nn.ReLU, nn.Hardswish)):
            setattr(mod, name, nn.Identity())
        elif isinstance(child, ConvLayer):
            c = child.conv
            if c.groups > 1 and c.groups == c.in_channels:
                setattr(mod, name, nn.Identity() if c.stride == (1, 1) else Subsample(c.stride[0]))
                continue
            if child.norm is not None and isinstance(child.norm, nn.BatchNorm2d):
                child.conv, child.norm = fold_bn(c, child.norm), None
            child.act = None
        elif isinstance(child, nn.Conv2d) and child.groups > 1 and child.groups == child.in_channels:
            setattr(mod, name, nn.Identity() if child.stride == (1, 1) else Subsample(child.stride[0]))
        else:
            strip_all(child)


def dense_mbconvs(mod, prefix=""):
    from efficientvit.models.nn.ops import MBConv, ResidualBlock
    n = 0
    for name, child in mod.named_children():
        if (isinstance(child, ResidualBlock) and isinstance(child.main, MBConv) and child.shortcut is not None
                and child.main.depth_conv.conv.stride == (1, 1)):
            setattr(mod, name, DenseMBConv(child))
            n += 1
        else:
            n += dense_mbconvs(child)
    return n


class Named(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, image):
        return self.m(image)


def export_encoder(enc, name):
    x = torch.rand(1, 3, 512, 512) * 255
    with torch.no_grad():
        enc(x)  # warm SeparableResize caches
    enc, x = enc.half().eval(), x.half()
    with torch.no_grad():
        y = enc(x)
    print(name, "out", tuple(y.shape), "finite", bool(torch.isfinite(y).all()))
    convert(Named(enc), [x], [], os.path.join(OUT, f"{name}.aimodel"))
    print("saved", name)


class MatMul(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.register_buffer("w", (torch.randn(n, n) / n ** 0.5).half())

    def forward(self, x):
        return x @ self.w


class Stream(nn.Module):
    def forward(self, x):
        return x * 1.5 + 0.5


def main():
    sam = load_sam("l0")
    bb = sam.image_encoder.backbone
    mods = dict(bb.named_modules())
    for tag, name, shape in [("s3", "stages.3.op_list.1", (1, 256, 32, 32)),
                             ("s4", "stages.4.op_list.1.local_module", (1, 512, 16, 16))]:
        convert(Named(DenseMBConv(mods[name]).half()), [torch.randn(*shape).half()], [],
                os.path.join(OUT, f"blk_{tag}_dense.aimodel"))
        print("saved blk", tag)

    enc = make_coreml_friendly(EncoderWrapper(load_sam("l0"))).eval()
    print("dense MBConv blocks:", dense_mbconvs(enc))
    export_encoder(enc, "enc_dense_mb")

    enc = make_coreml_friendly(EncoderWrapper(load_sam("l0"))).eval()
    strip_all(enc)
    export_encoder(enc, "enc_dense_all")

    convert(MatMul(4096), [torch.randn(4096, 4096).half()], [], os.path.join(OUT, "peak_matmul_4096.aimodel"))
    convert(Stream(), [torch.randn(64 * 1024 * 1024).half()], [], os.path.join(OUT, "peak_stream_64M.aimodel"))
    print("saved peaks")


if __name__ == "__main__":
    main()
