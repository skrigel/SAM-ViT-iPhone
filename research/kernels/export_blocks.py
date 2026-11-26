"""Export single-block Core AI models for block-level timing: stock fp16 block vs FusedMBConvBlock.
  USE_OS_COREAI=1 .venv-coreai/bin/python kernels/export_blocks.py -> work/models/blk_<s3|s4>_<stock|fused>.aimodel
"""
import copy, os, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
from coreai_kernel_util import convert  # noqa: E402
from fused_mbconv import KERNELS, FusedMBConvBlock  # noqa: E402
from evitsam import ROOT, load_sam  # noqa: E402

class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x):
        return self.m(x)

bb = load_sam("l0").image_encoder.backbone
mods = dict(bb.named_modules())
for tag, name, shape in [("s3", "stages.3.op_list.1", (1, 256, 32, 32)), ("s4", "stages.4.op_list.1.local_module", (1, 512, 16, 16))]:
    x = torch.randn(*shape).half()
    convert(Wrap(copy.deepcopy(mods[name]).half()), [x], [], os.path.join(ROOT, "models", f"blk_{tag}_stock.aimodel"))
    convert(Wrap(FusedMBConvBlock(mods[name])), [x], KERNELS, os.path.join(ROOT, "models", f"blk_{tag}_fused.aimodel"))
    print("exported", tag)
