"""Fixed-shape prompt decoder with the app's exact interface (drop-in for SAMPromptDecoder.mlpackage).

The app's decoder has flexible point_coords/point_labels shapes, which keeps Core ML from placing any of its 220 ops on
the GPU or ANE: it runs on the CPU at ~41 ms on an M5 Pro. With a fixed 2-point shape (the app always sends the two box
corners, labels 2 and 3) the same network runs on the GPU in ~2 ms (tools/decoder_placement.py).

Interface (same names/dtypes as the app's model):
  in:  image_embeddings (1,256,64,64) fp16, point_coords (1,2,2) fp16 in the 1024 frame, point_labels (1,2) fp16
  out: masks (1,4,256,256) fp16 low-res logits (index 0 = single-mask output, 1-3 = multimask), iou_predictions (1,4) fp16

  python tools/convert_decoder_app.py [--out ../iPhone-SAM-ViT/Models/SAMPromptDecoderFixed.mlpackage]
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evitsam import ROOT, load_sam  # noqa: E402


class AppDecoder(nn.Module):
    """SAM prompt encoder (points only, no mask input) + mask decoder, like segment_anything's SamOnnxModel."""

    def __init__(self, sam):
        super().__init__()
        self.pe = sam.prompt_encoder
        self.decoder = sam.mask_decoder
        self.register_buffer("dense_pe", sam.prompt_encoder.get_dense_pe())

    def forward(self, image_embeddings, point_coords, point_labels):
        coords = (point_coords + 0.5) / 1024.0
        emb = self.pe.pe_layer._pe_encoding(coords)                       # (1, N, 256)
        labels = point_labels.unsqueeze(-1)
        emb = emb * (labels != -1)
        emb = emb + self.pe.not_a_point_embed.weight * (labels == -1)
        for i in range(self.pe.num_point_embeddings):
            emb = emb + self.pe.point_embeddings[i].weight * (labels == i)
        dense = self.pe.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(1, -1, 64, 64)
        masks, iou = self.decoder.predict_masks(image_embeddings=image_embeddings, image_pe=self.dense_pe,
                                                sparse_prompt_embeddings=emb, dense_prompt_embeddings=dense)
        return masks, iou


def main():
    import coremltools as ct

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "SAMPromptDecoderFixed.mlpackage"))
    args = ap.parse_args()
    dec = AppDecoder(load_sam("l0")).eval()
    ex = (torch.randn(1, 256, 64, 64), torch.tensor([[[100.0, 100.0], [600.0, 500.0]]]), torch.tensor([[2.0, 3.0]]))
    traced = torch.jit.trace(dec, ex)
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image_embeddings", shape=(1, 256, 64, 64), dtype=np.float16),
                ct.TensorType(name="point_coords", shape=(1, 2, 2), dtype=np.float16),
                ct.TensorType(name="point_labels", shape=(1, 2), dtype=np.float16)],
        outputs=[ct.TensorType(name="masks", dtype=np.float16), ct.TensorType(name="iou_predictions", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )
    ml.short_description = ("EfficientViT-SAM-L0 prompt encoder + mask decoder, fixed 2-point shape (box corners, labels 2/3) "
                            "so it can run on the GPU/ANE. Same interface as SAMPromptDecoder.")
    ml.author = "converted from mit-han-lab/efficientvit (Apache-2.0); research/tools/convert_decoder_app.py"
    ml.save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
