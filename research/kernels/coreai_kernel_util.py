"""Helpers for developing Core AI inline Metal kernels (coreai_torch.TorchMetalKernel).

Conventions learned so far (kernels/smoke.py):
- Kernel parameters are Metal 4 `tensor<device T, dextents<int, R>, tensor_handle>`; coordinates are innermost-first,
  so a PyTorch (N, C, H, W) tensor is indexed t[w, h, c, n] and a 2-D (rows, cols) tensor t[col, row].
- Core AI compiles the Metal source when the model is specialized; no offline Metal toolchain is needed.

run_on_gpu(module, inputs) converts a small module with torch.export -> Core AI, specializes it for the GPU and runs it.
"""

import asyncio
import os
import pathlib
import shutil
import tempfile

import numpy as np
import torch

os.environ.setdefault("USE_OS_COREAI", "1")
import coreai.runtime as rt  # noqa: E402
from coreai_torch import TorchConverter, get_decomp_table  # noqa: E402

MPP_HEADER = "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp::tensor_ops;\n"


def convert(module, example_inputs, kernels, path):
    ep = torch.export.export(module.eval(), tuple(example_inputs)).run_decompositions(get_decomp_table())
    prog = TorchConverter().register_custom_kernels(kernels).add_exported_program(ep).to_coreai()
    path = pathlib.Path(path)
    shutil.rmtree(path, ignore_errors=True)
    prog.save_asset(path)
    return path


async def _run(path, inputs, unit):
    kind = {"gpu": rt.ComputeUnitKind.gpu, "cpu": rt.ComputeUnitKind.cpu, "ane": rt.ComputeUnitKind.neural_engine}[unit]()
    m = await rt.AIModel.load(path, rt.SpecializationOptions.from_preferred_compute_unit_kind(kind))
    f = m.load_function(m.function_names[0])
    def plain(x):  # hook captures can be tensor subclasses, which refuse .numpy()
        return np.array(x.detach().cpu().float().tolist(), dtype=np.float16 if x.dtype == torch.float16 else np.float32)
    feed = {n: rt.NDArray(np.ascontiguousarray(plain(x))) for n, x in zip(f.desc.input_names, inputs)}
    out = await f(feed)
    return [out[n].numpy() for n in f.desc.output_names]


def run_on_gpu(module, inputs, kernels, path=None, unit="gpu"):
    path = path or os.path.join(tempfile.mkdtemp(), "k.aimodel")
    convert(module, inputs, kernels, path)
    return asyncio.run(_run(path, inputs, unit))


def compare(name, got, ref):
    got, ref = np.asarray(got, np.float64).ravel(), np.asarray(ref, np.float64).ravel()
    err = np.abs(got - ref)
    cos = float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-30))
    rel = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-30))
    print(f"{name:40s} max_abs {err.max():.3e}  rel_l2 {rel:.3e}  cos {cos:.6f}  nan {int(np.isnan(got).sum())}")
    return dict(max_abs=float(err.max()), rel_l2=rel, cos=cos)
