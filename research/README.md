# research/ — model side

Experiments behind the encoder in `iPhone-SAM-ViT/Models/SAMImageEncoderANE.mlpackage`, plus the Core ML / Core AI /
MPSGraph comparison. Start with **PROGRESS.md** (findings). Then read PROGRAM.md (how experiments are run and logged), results.tsv
(every run), and tasks.json (what's next).

```sh
./init.sh                                            # venvs, checkpoints, COCO, Swift bench (big files go in work/)
P=.venv/bin/python
$P tools/convert.py --precision fp16 --gelu expform --gelu-where 'stages\.(3|4)' --encoder-only --fp16-output --name ane
$P tools/eval_coco.py --backend coreml --encoder work/models/EViTSAM_Encoder_ane.mlpackage --unit ane --tag my_run
USE_OS_COREAI=1 .venv-coreai/bin/python tools/convert_coreai.py --gelu expform --gelu-where 'stages\.(3|4)'
$P tools/runtime_matrix.py                           # latency + energy, Core ML / Core AI / MPSGraph
```

| tool | what it does |
|---|---|
| `tools/evitsam.py` | model loading + exact rewrites: `ExpFormGELU`, `ClenshawGELU`, `TanhFormGELU*`, linear-op resize, fp16-safe attention |
| `tools/convert.py` / `convert_coreai.py` | PyTorch → .mlpackage / .aimodel |
| `tools/eval_coco.py` | frozen evaluator (500 COCO val2017 images, GT boxes) |
| `tools/probe_ane_ops.py`, `tools/taps.py` | single-op numerics per compute unit; per-block cosine vs fp32 |
| `tools/eval_decoder.py` | the app's decoder per compute unit, mask-selection rules |
| `tools/train_distill.py` | ReLU student distillation (fixed wall-clock budget) |
| `bench/RuntimeBench.swift`, `tools/runtime_matrix.py` | latency + power/energy for Core ML, Core AI, MPSGraph |
