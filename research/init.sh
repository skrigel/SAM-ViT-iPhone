#!/usr/bin/env bash
# Rebuild the research environment from scratch (idempotent). Run from research/.
# Big artifacts go under work/ (git-ignored): checkpoints, COCO, converted models, students, logs.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p work/{checkpoints,data/coco,models,students,logs}
W=work

# EfficientViT source (SAM model definitions)
[ -d $W/third_party/efficientvit ] || git clone --depth 1 https://github.com/mit-han-lab/efficientvit.git $W/third_party/efficientvit

# Env 1: coremltools 9 (needs torch 2.7, numpy < 2.5). Conversion to Core ML, evals, probes, distillation.
[ -d .venv ] || uv venv -q -p 3.12 .venv
VIRTUAL_ENV=.venv uv pip install -q -r requirements.txt
# Env 2: Core AI (coreai-torch needs torch >= 2.8). Conversion to .aimodel, Core AI evals.
[ -d .venv-coreai ] || uv venv -q -p 3.12 .venv-coreai
VIRTUAL_ENV=.venv-coreai uv pip install -q -r requirements-coreai.txt

# Checkpoints
for m in l0 l2 xl0; do
  f=$W/checkpoints/efficientvit_sam_$m.pt
  [ -f $f ] || curl -sSL -o $f https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_$m.pt
done

# COCO 2017: val (eval) + annotations; train (distillation only)
cd $W/data/coco
[ -d val2017 ] || { curl -sSLO http://images.cocodataset.org/zips/val2017.zip && unzip -q val2017.zip; }
[ -d annotations ] || { curl -sSLO http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip -q annotations_trainval2017.zip; }
[ -d train2017 ] || { curl -sSLO http://images.cocodataset.org/zips/train2017.zip && unzip -q train2017.zip; }
cd -

# Swift runtime benchmark (needs Xcode 27; the Core AI runtime ships with macOS/iOS 27)
DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer} xcrun swiftc -O -parse-as-library bench/RuntimeBench.swift -o $W/runtimebench
echo "ready. Core AI scripts need USE_OS_COREAI=1."
