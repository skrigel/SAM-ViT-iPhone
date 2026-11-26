"""Fused MBConv for EfficientViT-SAM as two Core AI inline Metal kernels (MetalPerformancePrimitives matmul2d).

Stock MBConv (stride 1, identity residual):  x -> 1x1 expand + bias -> GELU -> 3x3 depthwise + bias -> GELU
                                              -> 1x1 project -> BN -> + x
The expanded tensor (4-6x the channels of x) goes through memory five times in an unfused graph
(expand out, GELU in/out, depthwise in/out, GELU in/out, project in).

K1 `mbconv_expand`:  E = GELU(W1 @ X + b1)                    matmul with bias+GELU in the cooperative-tensor epilogue
K2 `mbconv_dw_proj`: Y = W3' @ GELU(dw3x3(E) + b2) + b3' + X  the depthwise+GELU result is built KC channels at a time in
                                                              threadgroup memory and fed straight into the project matmul
                                                              (multiply-accumulate); BN folded into W3'/b3'; residual in the epilogue
So E is written once and read once; the depthwise output never touches device memory.
All matmuls accumulate in fp32; tensors in device memory are fp16.

Layouts: activations are 2-D (C, H*W) row-major, which Metal sees as extents (H*W, C) (innermost first).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from coreai.authoring import MetalParameter
from coreai_torch._torch_metal_kernel import TorchMetalKernel

from coreai_kernel_util import MPP_HEADER

# Tiles. K1: TM output channels x TN pixels per threadgroup. K2: TMO output channels x TP pixels, KC expanded channels per step.
TM, TN = 64, 32
TMO, TP, KC = 64, 32, 32
THREADS = 128  # 4 SIMD-groups

GELU_HELPERS = MPP_HEADER + """
#ifndef SAM_GELU_TANH
#define SAM_GELU_TANH
inline float gelu_tanh(float v) {
    float u = 0.7978845608f * (v + 0.044715f * v * v * v);
    return 0.5f * v * (1.0f + metal::precise::tanh(u));
}
#endif
"""


def _expand_ref(w1: torch.Tensor, x: torch.Tensor, b1: torch.Tensor) -> torch.Tensor:
    return F.gelu(w1.float() @ x.float() + b1.float()[:, None], approximate="tanh").to(x.dtype)


def _dw_proj_ref(e: torch.Tensor, dw: torch.Tensor, b2: torch.Tensor, w3: torch.Tensor, b3: torch.Tensor,
                 x: torch.Tensor, width: int) -> torch.Tensor:
    c, hw = e.shape
    d = F.conv2d(e.float().reshape(1, c, hw // width, width), dw.float().reshape(c, 1, 3, 3), b2.float(), padding=1, groups=c)
    d = F.gelu(d, approximate="tanh").reshape(c, hw)
    return (w3.float() @ d + b3.float()[:, None] + x.float()).to(x.dtype)


K1_SRC = f"""
    constexpr auto desc = matmul2d_descriptor({TM}, {TN}, static_cast<int>(dynamic_extent));
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mA = w1.slice(0, tgid.y * {TM});
    auto mB = x.slice(tgid.x * {TN}, 0);
    auto cT = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
    op.run(mA, mB, cT);
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {{
        if (cT.is_valid_element(i)) {{
            auto idx = cT.get_multidimensional_index(i);
            int row = tgid.y * {TM} + idx[1], col = tgid.x * {TN} + idx[0];
            e[col, row] = (half)gelu_tanh(cT[i] + (float)b1[row]);
        }}
    }}
"""

K2_SRC = f"""
    threadgroup half dbuf[{KC * TP}];
    tensor<threadgroup half, extents<int, {TP}, {KC}>, tensor_inline> dt(dbuf, extents<int, {TP}, {KC}>());
    constexpr auto desc = matmul2d_descriptor({TMO}, {TP}, {KC}, false, false, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    const int p0 = tgid.x * {TP}, r0 = tgid.y * {TMO};
    const int cexp = e.get_extent(1), hw = e.get_extent(0), height = hw / width;
    auto a0 = w3.slice<{KC}, {TMO}>(0, r0);
    auto cT = op.get_destination_cooperative_tensor<decltype(a0), decltype(dt), float>();
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
    for (int k0 = 0; k0 < cexp; k0 += {KC}) {{
        for (int j = (int)tid; j < {KC * TP}; j += {THREADS}) {{
            const int kk = j / {TP}, pp = j % {TP};
            const int c = k0 + kk, p = p0 + pp, h = p / width, w = p % width;
            float acc = (float)b2[c];
            #pragma unroll
            for (int dy = -1; dy <= 1; ++dy) {{
                const int hh = h + dy;
                if (hh < 0 || hh >= height) continue;
                #pragma unroll
                for (int dx = -1; dx <= 1; ++dx) {{
                    const int ww = w + dx;
                    if (ww < 0 || ww >= width) continue;
                    acc += (float)dw[(dy + 1) * 3 + (dx + 1), c] * (float)e[hh * width + ww, c];
                }}
            }}
            dbuf[kk * {TP} + pp] = (half)gelu_tanh(acc);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA = w3.slice<{KC}, {TMO}>(k0, r0);
        op.run(mA, dt, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {{
        if (cT.is_valid_element(i)) {{
            auto idx = cT.get_multidimensional_index(i);
            int row = r0 + idx[1], col = p0 + idx[0];
            y[col, row] = (half)(cT[i] + (float)b3[row] + (float)x[col, row]);
        }}
    }}
"""

K1 = TorchMetalKernel(
    name="mbconv_expand", input_names=["w1", "x", "b1"], result_names=["e"], src=K1_SRC, torch_defn=_expand_ref,
    helper_src=GELU_HELPERS,
    metal_params=[MetalParameter(name="tgid", dtype="uint2", attr="threadgroup_position_in_grid")],
)
K2 = TorchMetalKernel(
    name="mbconv_dw_proj", input_names=["e", "dw", "b2", "w3", "b3", "x", "width"], result_names=["y"], src=K2_SRC,
    torch_defn=_dw_proj_ref, helper_src=GELU_HELPERS,
    metal_params=[MetalParameter(name="tgid", dtype="uint2", attr="threadgroup_position_in_grid"),
                  MetalParameter(name="tid", dtype="uint", attr="thread_index_in_threadgroup")],
)
KERNELS = [K1, K2]


class FusedMBConvBlock(nn.Module):
    """Drop-in for ResidualBlock(main=MBConv(stride 1), shortcut=IdentityLayer), fp16, batch 1."""

    def __init__(self, block):
        super().__init__()
        m = block.main
        inv, dwc, pc = m.inverted_conv.conv, m.depth_conv.conv, m.point_conv
        assert dwc.stride == (1, 1) and dwc.kernel_size == (3, 3) and block.shortcut is not None
        cexp = inv.out_channels
        bn = pc.norm
        g = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        with torch.no_grad():
            self.register_buffer("w1", inv.weight[:, :, 0, 0].clone().half())
            self.register_buffer("b1", inv.bias.clone().half())
            self.register_buffer("dw", dwc.weight.reshape(cexp, 9).clone().half())
            self.register_buffer("b2", dwc.bias.clone().half())
            self.register_buffer("w3", (pc.conv.weight[:, :, 0, 0] * g[:, None]).half())
            self.register_buffer("b3", (bn.bias - bn.running_mean * g).half())
        self.cin, self.cexp, self.cout = inv.in_channels, cexp, pc.conv.out_channels
        for n, d in [("cexp", cexp), ("cout", self.cout)]:
            assert d % 64 == 0, f"{n}={d} must be a multiple of 64"

    def forward(self, x):
        _, c, h, w = x.shape
        hw = h * w
        assert hw % 32 == 0
        x2 = x.reshape(c, hw)
        e = K1(self.w1, x2, self.b1, threads_per_grid=(hw // TN * THREADS, self.cexp // TM, 1),
               threads_per_thread_group=(THREADS, 1, 1), result_shapes=[[self.cexp, hw]])
        y = K2(e, self.dw, self.b2, self.w3, self.b3, x2, w, threads_per_grid=(hw // TP * THREADS, self.cout // TMO, 1),
               threads_per_thread_group=(THREADS, 1, 1), result_shapes=[[self.cout, hw]])
        return y.reshape(1, self.cout, h, w)


def fuse_mbconvs(encoder, where=lambda name: True):
    """Replace every eligible MBConv residual block (stride 1, identity shortcut) in-place. Returns replaced names."""
    from efficientvit.models.nn.ops import MBConv, ResidualBlock
    done = []

    def visit(mod, prefix):
        for name, child in mod.named_children():
            q = f"{prefix}.{name}" if prefix else name
            if (isinstance(child, ResidualBlock) and isinstance(child.main, MBConv) and child.shortcut is not None
                    and child.main.depth_conv.conv.stride == (1, 1) and where(q)):
                setattr(mod, name, FusedMBConvBlock(child))
                done.append(q)
            else:
                visit(child, q)

    visit(encoder, "")
    return done
