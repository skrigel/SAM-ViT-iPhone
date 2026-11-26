# Research program: model side of SAM-ViT-iPhone

These are the rules for anyone, human or agent, who runs experiments in `research/`.
They combine two published agent harnesses:

- Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) `program.md`: a fixed evaluator nobody edits during a search,
  a fixed budget per experiment, keep-or-revert on the score, one `results.tsv` row per run, and a preference for simple changes.
- Anthropic's [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents):
  an `init.sh` that rebuilds the environment, a progress file and git log that let a fresh session pick up where the last one stopped,
  and a JSON task list that is changed only by flipping `status`.

## Goals (from the project proposal)

1. **FP16 on the Neural Engine.** The stock EfficientViT-SAM-L0 encoder is lossless in fp16 on the GPU and wrong on the ANE.
   Target: at most 0.5 mIoU below fp32 on the fixed COCO eval, with encoder latency at or below the fp16 GPU encoder (24 ms on A19 Pro).
2. **Runtime comparison.** Compare Core ML, Core AI (the iOS/macOS 27 successor to Core ML) and direct Metal (MPSGraph) on latency,
   accuracy and placement for the same encoder.

## Files

| File | Who edits it | Purpose |
|---|---|---|
| `PROGRAM.md` | humans | These rules. |
| `tasks.json` | agent flips `status` and adds `notes` only | The task list. Do not delete or reword tasks; add new ones at the end. |
| `PROGRESS.md` | agent appends, newest first | What was done, what was learned, what is next. Read it before starting work. |
| `results.tsv` | agent appends | One row per experiment, including failures. |
| `tools/eval_coco.py` | **frozen** once the baseline row exists | The evaluator. Changing it invalidates every row in `results.tsv`. |
| `init.sh` | humans/agent | Rebuilds the Python environments and downloads checkpoints and data. |

## The evaluator

`tools/eval_coco.py` scores an encoder+decoder pair on a fixed 500-image subset of COCO val2017
(seed 0, one annotation per image, area ≥ 1% of the image, ground-truth box as the prompt) and reports mIoU against the
ground-truth masks and the embedding cosine against PyTorch fp32. Images 500-999 of the same shuffle are a separate dev split used for quick
checks. Training never sees val2017.

## Loop

1. Start of a session: `pwd`, `git log --oneline -15`, read `PROGRESS.md` and `tasks.json`, pick the highest-priority task that is not done.
2. State a hypothesis in one line before running anything.
3. Run the cheapest test that could falsify it first (a single-op or single-stage probe before a full eval, the dev split before the full split).
4. Log every run to `results.tsv`: `date  commit  experiment  unit  precision  mIoU  cos  latency_ms  status  note`.
   `status` is `keep`, `discard` or `crash`.
5. **Keep or revert.** A change is kept only if it improves the target metric on the fixed evaluator without breaking another unit's accuracy.
   Otherwise revert it and record why in the `note` column. A tiny gain that adds a lot of complexity is a `discard`.
6. Commit kept changes with a message that says what changed and what it measured. Append a `PROGRESS.md` entry.
7. Do not stop to ask whether to continue. Stop only for a decision that is the user's to make, and record it under "Open questions".

## Budgets

- Probes and single-stage tests: under 2 minutes.
- Full Core ML / Core AI eval of one model on one unit: about 5 minutes for 500 images.
- Distillation runs: one fixed wall-clock budget per run (set in the run's command line and logged), so runs are comparable.
  Long runs go in the background and log to `work/logs/`.

## Measurement hygiene

- Mac numbers come from an M5 Pro (same GPU/ANE generation as the A19 Pro). They are proxies; the iPhone 17 Pro numbers need the app.
- Latency: median of ≥ 50 runs after ≥ 5 warm-up runs, same input, nothing else heavy running. Report p10/p90 too.
- Never compare numbers from different evaluator versions or different image sets.
