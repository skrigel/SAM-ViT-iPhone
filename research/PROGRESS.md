# Progress log (newest first)

## 2026-11-26 — session 1: fp16 on the Neural Engine; Core ML vs Core AI vs Metal

**Setup.** Mac with an M5 Pro (same GPU/ANE generation as the A19 Pro), macOS 27.2, Xcode 27.2. Core AI is in the
macOS/iOS 27 SDK. Apple's converter is `coreai-torch` 0.4.4 (PyPI), which needs torch ≥ 2.8, so it has its own venv.
The Core AI runtime specializes models on-device. That works without the Metal Toolchain download that the `aimodelc`
CLI asks for. No iPhone was attached, so **all numbers are from the Mac**; `bench/RuntimeBench.swift` only uses APIs that exist
on iOS 27 and can go into an app target for iPhone numbers.

**Evaluator** (`tools/eval_coco.py`, frozen): 500 COCO val2017 images, one GT box prompt each, mIoU vs GT masks.
The decoder is always PyTorch fp32, so only the encoder varies. Dev split: the next 500 images. Paired standard errors
against the fp32 per-image IoUs are ~0.1 for good models.

### Result: fp16 works on the Neural Engine, no retraining

| L0 encoder, fp16 | unit | mIoU (fp32 = 83.56) | Δ | emb. cosine | p50 latency | energy / frame |
|---|---|---|---|---|---|---|
| Core ML, stock conversion | ANE | 50.16 | −33.40 | 0.217 | 8.0 ms | 81 mJ |
| **App's bundled `SAMImageEncoder`** | ANE | **4.46** | −79.10 | 0.093 | — | — |
| Core ML, stock | GPU | 83.46 | −0.10 | 0.9996 | 4.7 ms | 207 mJ |
| **Core ML, ANE-safe (ExpForm GELU in stages 3–4 + linear resize)** | **ANE** | **83.46** | **−0.10 ± 0.09** | 0.9985 | **13.2 ms** | 134 mJ |
| **Core AI, same ANE-safe model** | **ANE** | **83.69** | **+0.13** | 0.9984 | **10.9 ms** | **90 mJ** |
| Core AI, same model | GPU | — | — | — | 4.6 ms | 210 mJ |

Both new models are in the app; the default placement is unchanged. The ANE-safe encoder is `iPhone-SAM-ViT/Models/SAMImageEncoderANE.mlpackage` (same I/O as
`SAMImageEncoder`). Opt in with `segmenter.load(encoderUnits: .cpuAndNeuralEngine, encoderURL: Segmenter.aneEncoderURL)`.
On the Mac it uses 1.6–2.3× less energy per frame than the GPU path at 2–3× the latency. Whether that trade is right
for the iPhone needs the on-device numbers.

### Why stock fp16 fails on the ANE: two independent bugs

Fixing either one alone gives 57.7 or 22.1 mIoU.

1. **The ANE's built-in GELU is inaccurate.** In single-op probes (`tools/probe_ane_ops.py`) its max error is 6.2e-3,
   largest near 0, for both the erf and tanh forms (CPU/GPU: 6e-4). Its sigmoid and tanh are also LUT-grade
   (2–3e-3). Its ReLU, add, mul, exp and divide are accurate. The high-gain folded BatchNorms in stage 3 (up to 321×)
   amplify the GELU error. The per-block taps (`tools/taps.py`) show the ANE diverging from stage 3 block 1
   (cos 0.87 → 0.38 by the end of stage 3). Stage 3 GELUs must be fixed; stage 4 + neck alone gives 58.0.
   *Not* the cause: the ANE's conv accumulation. On a 1×1 conv whose large terms cancel, the ANE is 20× more
   accurate than the CPU/GPU fp16 paths.
2. **The ANE compiler miscompiles the neck's bicubic resize** when it is written as `matmul(A_h, x) @ A_w^T` feeding the
   neck's sum. With every GELU fixed, the backbone matches fp32 (cos 0.99) but the output has cos 0.19. Exposing that tensor
   as an extra model output made it correct, which pointed to a fusion bug. Writing the same resize as two `linear` ops
   fixes it (`evitsam.SeparableResize`).

### GELU rewrites tried (all exact identities; `tools/evitsam.py`)

| form | ops | ANE max err | where | ANE ms | full-eval Δ | big losses* |
|---|---|---|---|---|---|---|
| built-in `gelu` | 1 | 6.2e-3 | all | 8.0 | −33.4 (with linear resize: −25.9) | 442 |
| `ClenshawGELU` relu(x) − Chebyshev g(\|x\|) | ~27 | 1.0e-3 | all | 36.2 | −0.19 | 3 |
| `ClenshawGELU` | ~27 | | stages 3–4 | 25.2 | (dev −0.10) | |
| `TanhFormGELU2` 0.5x + 0.5x·tanh(u) | ~7 | 1.8e-3 | stages 3–4 | 11.7 | −0.55 | 15 |
| **`ExpFormGELU` relu(x) − t/(1+exp(2u(t)))** | ~10 | **1.0e-3** (rms 2.9e-4) | **stages 3–4** | **13.2** | **−0.10** | **2** |
| `ExpFormGELU` | ~10 | | all | 17.1 | −0.12 | |

\* images losing > 10 IoU points vs fp32.

### Runtimes: Core ML vs Core AI vs MPSGraph (Metal), M5 Pro, `tools/runtime_matrix.py`

| encoder fp16 (stock) | GPU p50 | ANE p50 | first inference (GPU) | GPU energy |
|---|---|---|---|---|
| Core ML | 4.68 ms | 7.99 ms | 241 ms | 207 mJ |
| Core AI | 4.33 ms | 7.22 ms | 95 ms | 207 mJ |
| MPSGraph from the .mlpackage (Metal) | 4.31 ms | — | 838 ms | 206 mJ |

- **On the GPU it is a tie.** Core AI's profiler shows one `call.MPSGraph` per inference, so Core AI and the Core ML GPU path both
  run on MPSGraph. Hand-driving MPSGraph removes no measurable overhead; it only adds JIT compile at first run (0.8–1.4 s,
  nothing cached). A real "Metal" win would need custom kernels. Core AI exposes those through `TorchMetalKernel` in
  `coreai-torch`; not tried yet.
- **On the ANE, Core AI is faster:** stock 7.2 vs 8.0 ms; ANE-safe 10.9 vs 13.2 ms at 90 vs 134 mJ. It also loads from its
  cache in about 0.1 s.
- **Core AI caveats:** (a) Swift `AIModel(contentsOf:)` on the ANE fails intermittently with `Foundation nilError`; the Clenshaw
  model failed 5/5 retries while the same file loads from Python. `RuntimeBench` retries and records `load_attempts`.
  (b) fp32 Core AI on the GPU is slower than Core ML (29 vs 10.6 ms). (c) Building a Core AI `NDArray` from a Swift array
  each frame cost ~30 ms. Build it once and reuse it (or wrap a camera `CVPixelBuffer`).
- The ANE-safe GELUs cost nothing on the GPU through Core AI or MPSGraph (4.6 ms), so one model can serve both units.

### Decoder (`tools/eval_decoder.py`, `tools/decoder_placement.py`, `tools/convert_decoder_app.py`)
- The app's `SAMPromptDecoder` runs **entirely on the CPU** even with `.all`: all 220 ops, per `MLComputePlan`. Its
  flexible-shape point inputs block GPU/ANE placement. It takes **43 ms per decode** on the M5 Pro, more than the encoder.
- `SAMPromptDecoderFixed.mlpackage` (now in the app) is the same network with a fixed 2-point shape and the same
  interface. It runs in **1.6 ms on the GPU**, 2.4 ms on the ANE, or 5.0 ms on the CPU. Accuracy on 200 dev images
  (argmax-mask mIoU): app decoder 85.92; fixed 85.90 on CPU/GPU and 86.07 on the ANE. On the ANE its mask-0
  agreement with torch is lower (0.985, min 0.74 vs 0.999 on GPU), so use the GPU for it:
  `load(decoderUnits: .cpuAndGPU, decoderURL: Segmenter.fixedDecoderURL)`.
- The app picks the mask with the highest predicted IoU. For box prompts that beats SAM's recommended single mask
  by +1.8 mIoU (85.92 vs 84.07), so keep it.

### ReLU distillation (proposal 4.1)
`tools/train_distill.py`: GELU → ReLU everywhere, L2 to the teacher embedding, COCO train2017, batch 8, fixed wall-clock budget
on the M5 Pro GPU. Lessons so far:
- BatchNorm in train mode at batch 8: the training-batch cosine reached 0.97, but in inference mode the student scored
  72.97 mIoU on dev with cosine 0.76 (it learned to lean on batch statistics). Discarded.
- Frozen BN with the teacher's running stats: the swap changes every BN input distribution, so inference cosine sat at
  ~0.53 and training went NaN at lr 1e-4. Crashed.
- **Recipe that trains:** re-estimate BN running stats on 200 batches after the swap, then freeze BN, lr 3e-5, grad clip 1.0.
  Inference-mode cosine 0.56 → 0.90 in 400 steps (~3.5 min).
- That 120-minute run only got through **790 steps**: the Mac went into idle sleep at 01:37 and the wall-clock budget
  ran out while it slept. Result on the dev split: **78.43 mIoU (−4.83 vs fp32), embedding cosine 0.932**, still climbing.
  It needs a rerun with the machine kept awake (`caffeinate -i`): ~13k steps fit in 2 h.
- **Rerun awake (caffeinate, 120 min, 13.4k steps, final loss 5.7e-4):**

  | ReLU student | split | mIoU | Δ vs fp32 | cos | ANE p50 |
  |---|---|---|---|---|---|
  | torch fp32 | dev 150 | 82.28 | −0.99 | 0.974 | |
  | torch fp32 | eval 500 | 82.55 | −1.00 | 0.973 | |
  | Core ML fp16, ANE | eval 500 | 81.02 | −2.54 | 0.953 | 8.34 ms |
  | *ExpForm s34 (no training), Core ML ANE* | eval 500 | 83.46 | −0.10 | 0.998 | 13.2 ms |

  The student is fast on the ANE (stock latency, no GELU at all), but it loses 1 mIoU to distillation and another
  1.5 to ANE fp16 rounding. Its stage-3 BNs still have high gains, so fp16 error is amplified there, as with GELU.
  Verdict: ExpForm stays the default. ReLU would need longer training plus fp16-aware fine-tuning (or BN gain
  regularisation) to be competitive; that is a 5–37% latency win (8.3 vs 13.2 Core ML / 10.9 Core AI) for −2.4 mIoU now.

### Next
1. Run `RuntimeBench` and the ANE model on the iPhone 17 Pro (latency, energy, thermal); decide GPU vs ANE.
2. ReLU student: optional fp16-aware fine-tune (train through `.half()` simulation) to close the 1.5-point ANE gap.
3. On device: the opt-in line in `RunView.prepare()` switches to the ANE encoder + GPU fixed decoder (builds for iOS 27 Simulator).
4. XL0 with the ExpForm fix. L2 is done: stock fp16 on the ANE scores 9.5 mIoU, ExpForm in stages 3–4 scores 83.65
   (fp32 83.79, dev split), at 22.6 ms on the ANE vs 8.0 ms stock on the GPU.

### Open questions for the team
- The repo's Xcode project fails to build from the command line (`Multiple commands produce .../Info.plist`):
  `iPhone-SAM-ViT/Info.plist` sits in the synchronized folder, so it is both copied as a resource and generated.
  It needs a membership exception or `INFOPLIST_FILE`. Left alone because it is a project setting.
