"""Convert EfficientViT-SAM to Core ML as two models: image encoder + prompt/mask decoder.

Usage: python tools/convert.py --model l0 [--precision fp16|fp32]
Outputs: models/EViTSAM_<model>_Encoder_<prec>.mlpackage, models/EViTSAM_Decoder_<model>_<prec>.mlpackage
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evitsam import INPUT_SIZE, ROOT, ClenshawGELU, ExpFormGELU, TanhFormGELU, TanhFormGELU2, load_sam, make_coreml_friendly, replace_gelu  # noqa: E402

MEAN = [123.675, 116.28, 103.53]
STD = [58.395, 57.12, 57.375]


class EncoderWrapper(nn.Module):
    """Takes RGB pixels in [0, 255] (Core ML ImageType), normalizes inside the graph."""

    def __init__(self, sam):
        super().__init__()
        self.encoder = sam.image_encoder
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, image):
        return self.encoder((image - self.mean) / self.std)


class DecoderWrapper(nn.Module):
    """Box (+ optional previous-frame mask) -> single low-res mask logits + IoU.

    point_coords: (1, 2, 2) box corners in the 1024-long-side frame, labels fixed to (2, 3).
    mask_input:   (1, 1, 256, 256) previous low-res logits, gated by has_mask_input (1,).
    """

    def __init__(self, sam):
        super().__init__()
        self.pe = sam.prompt_encoder
        self.decoder = sam.mask_decoder
        self.register_buffer("dense_pe", sam.prompt_encoder.get_dense_pe())

    def forward(self, image_embeddings, box_coords, mask_input, has_mask_input):
        coords = (box_coords + 0.5) / 1024.0
        sparse = self.pe.pe_layer._pe_encoding(coords)
        sparse = torch.cat(
            [sparse[:, 0:1] + self.pe.point_embeddings[2].weight, sparse[:, 1:2] + self.pe.point_embeddings[3].weight],
            dim=1,
        )
        h = has_mask_input.view(1, 1, 1, 1)
        dense = h * self.pe.mask_downscaling(mask_input) + (1 - h) * self.pe.no_mask_embed.weight.reshape(1, -1, 1, 1)
        masks, iou = self.decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.dense_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
        )
        return masks[:, 0:1], iou[:, 0:1]


def main():
    import coremltools as ct

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="l0", choices=list(INPUT_SIZE))
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--gelu", default="native", choices=["native", "clenshaw", "tanhform", "tanhform2", "expform"],
                    help="clenshaw: ANE-accurate GELU built from relu/add/mul (evitsam.ClenshawGELU)")
    ap.add_argument("--gelu-where", default="", help="regex on module names to limit the GELU swap (default: all)")
    ap.add_argument("--gelu-degree", type=int, default=8)
    ap.add_argument("--student", help="pickled encoder nn.Module (pixels in, embedding out) to convert instead")
    ap.add_argument("--name", help="output name suffix (default: <model>_<precision>[_<gelu>])")
    ap.add_argument("--encoder-only", action="store_true")
    ap.add_argument("--tensor-input", action="store_true", help="float (1,3,S,S) pixel tensor input instead of an image")
    ap.add_argument("--fp16-output", action="store_true", help="fp16 embedding output, like the app's SAMImageEncoder")
    args = ap.parse_args()

    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    sam = load_sam(args.model)
    size = INPUT_SIZE[args.model]
    prec = ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32
    tag = args.name or f"{args.model}_{args.precision}" + ("" if args.gelu == "native" else f"_{args.gelu}")

    enc = torch.load(args.student, weights_only=False) if args.student else EncoderWrapper(sam)
    enc = make_coreml_friendly(enc).eval()
    if args.gelu != "native":
        import re
        make = {"clenshaw": lambda g: ClenshawGELU(g.approximate, args.gelu_degree),
                "tanhform": lambda g: TanhFormGELU(), "tanhform2": lambda g: TanhFormGELU2(),
                "expform": lambda g: ExpFormGELU()}[args.gelu]
        done = replace_gelu(enc, make,
                            where=lambda n: re.search(args.gelu_where, n) is not None)
        print(f"replaced {len(done)} GELUs")
    example = torch.rand(1, 3, size, size) * 255
    with torch.no_grad():
        enc(example)  # warm the SeparableResize matrix cache
    traced = torch.jit.trace(enc, example)
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=(1, 3, size, size)) if args.tensor_input
                else ct.ImageType(name="image", shape=(1, 3, size, size), color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name="image_embeddings", dtype=np.float16) if args.fp16_output
                 else ct.TensorType(name="image_embeddings")],
        compute_precision=prec,
        minimum_deployment_target=ct.target.iOS18,
    )
    ml.short_description = f"EfficientViT-SAM-{args.model.upper()} image encoder ({size}x{size})"
    ml.save(os.path.join(ROOT, "models", f"EViTSAM_Encoder_{tag}.mlpackage"))

    if args.encoder_only:
        print("saved", tag)
        return
    dec = DecoderWrapper(sam).eval()
    ex = (
        torch.randn(1, 256, 64, 64),
        torch.tensor([[[100.0, 100.0], [600.0, 500.0]]]),
        torch.zeros(1, 1, 256, 256),
        torch.zeros(1),
    )
    traced = torch.jit.trace(dec, ex)
    ml = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="image_embeddings", shape=(1, 256, 64, 64)),
            ct.TensorType(name="box_coords", shape=(1, 2, 2)),
            ct.TensorType(name="mask_input", shape=(1, 1, 256, 256)),
            ct.TensorType(name="has_mask_input", shape=(1,)),
        ],
        outputs=[ct.TensorType(name="low_res_masks"), ct.TensorType(name="iou")],
        compute_precision=prec,
        minimum_deployment_target=ct.target.iOS18,
    )
    ml.short_description = f"EfficientViT-SAM-{args.model.upper()} prompt encoder + mask decoder (box prompt)"
    ml.save(os.path.join(ROOT, "models", f"EViTSAM_Decoder_{tag}.mlpackage"))
    print("saved", tag)


if __name__ == "__main__":
    main()
