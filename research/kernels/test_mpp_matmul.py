"""Can a Core AI inline kernel use MetalPerformancePrimitives matmul2d? C[M,N] = A[M,K] @ B[K,N] + bias, then GELU.

Shapes = stage-3 MBConv expand: A = W1 (1024 x 256), B = X (256 x 1024 pixels).
"""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from coreai.authoring import MetalParameter  # noqa: E402
from coreai_kernel_util import MPP_HEADER, compare, run_on_gpu  # noqa: E402
from coreai_torch._torch_metal_kernel import TorchMetalKernel  # noqa: E402

TM, TN = 64, 32  # output tile per threadgroup: TM rows (out channels) x TN cols (pixels)


def ref(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(a.float() @ b.float() + bias.float()[:, None], approximate="tanh").to(a.dtype)


SRC = f"""
    // a: (M x K) extents (K, M); b: (K x N) extents (N, K); bias: (M) ; out: (M x N) extents (N, M)
    constexpr auto desc = matmul2d_descriptor({TM}, {TN}, static_cast<int>(dynamic_extent));
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mA = a.slice(0, tgid.y * {TM});
    auto mB = b.slice(tgid.x * {TN}, 0);
    auto cT = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
    op.run(mA, mB, cT);
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {{
        if (cT.is_valid_element(i)) {{
            auto idx = cT.get_multidimensional_index(i);   // (col, row) in the local tile
            int row = tgid.y * {TM} + idx[1], col = tgid.x * {TN} + idx[0];
            float v = cT[i] + (float)bias[row];
            float u = 0.7978845608f * (v + 0.044715f * v * v * v);
            out[col, row] = (half)(0.5f * v * (1.0f + metal::precise::tanh(u)));   // fp32 accumulate, one fp16 rounding
        }}
    }}
"""

K = TorchMetalKernel(
    name="mm_bias_gelu", input_names=["a", "b", "bias"], result_names=["out"], src=SRC, torch_defn=ref,
    helper_src=MPP_HEADER,
    metal_params=[MetalParameter(name="tgid", dtype="uint2", attr="threadgroup_position_in_grid")],
)


class M(torch.nn.Module):
    def forward(self, a, b, bias):
        m, n = a.shape[0], b.shape[1]
        return K(a, b, bias, threads_per_grid=(n // TN * 128, m // TM, 1), threads_per_thread_group=(128, 1, 1),
                 result_shapes=[[m, n]])


if __name__ == "__main__":
    torch.manual_seed(0)
    a = (torch.randn(1024, 256) * 0.06).half()
    b = torch.randn(256, 1024).half()
    bias = (torch.randn(1024) * 0.1).half()
    (out,) = run_on_gpu(M(), [a, b, bias], [K])
    compare("mm+bias+gelu vs torch fp32", out, ref(a.float(), b.float(), bias.float()))
