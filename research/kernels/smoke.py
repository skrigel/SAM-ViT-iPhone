"""Smoke test: a Core AI TorchMetalKernel inside a tiny model, converted and run on the GPU.
Learns the index order of Metal 4 tensor<> parameters (which coordinate is innermost)."""
import asyncio, os, pathlib, shutil, sys
import numpy as np, torch
from coreai_torch import TorchConverter, get_decomp_table
from coreai_torch._torch_metal_kernel import TorchMetalKernel
from coreai.authoring import MetalParameter
import coreai.runtime as rt

def ref(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0

# Write out[n,c,h,w] = n*1000 + c*100 + h*10 + w, indexing the tensor with 4 coordinates, to see the order.
k = TorchMetalKernel(
    name="probe_index", input_names=["x"], result_names=["out"],
    src="""
    // gid.x -> w, gid.y -> h, gid.z -> c (n = 0)
    out[gid.x, gid.y, gid.z, 0] = (half)(gid.z * 100 + gid.y * 10 + gid.x);
    """,
    torch_defn=ref,
    metal_params=[MetalParameter(name="gid", dtype="uint3", attr="thread_position_in_grid")]
,
)

class M(torch.nn.Module):
    def forward(self, x):
        return k(x, threads_per_grid=(5, 4, 3), threads_per_thread_group=(5, 4, 1), result_shapes=[[1, 3, 4, 5]])

x = torch.zeros(1, 3, 4, 5, dtype=torch.float16)
ep = torch.export.export(M(), (x,)).run_decompositions(get_decomp_table())
conv = TorchConverter()
from coreai_torch import _torch_metal_kernel  # noqa
if hasattr(conv, "register_custom_kernels"):
    conv.register_custom_kernels([k])
prog = conv.add_exported_program(ep).to_coreai()
p = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/probe.aimodel"); shutil.rmtree(p, ignore_errors=True)
prog.save_asset(p)

async def run():
    m = await rt.AIModel.load(p, rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
    f = m.load_function(m.function_names[0])
    out = await f({f.desc.input_names[0]: rt.NDArray(x.numpy())})
    return list(out.values())[0].numpy()
y = asyncio.run(run())
print(y.shape); print(y[0, :, :, :].astype(int))
