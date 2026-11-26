"""Latency + power for each (runtime, unit, model) via work/runtimebench, sampling power with `macmon pipe` (no sudo).

  python tools/runtime_matrix.py [--duration 8] [--only REGEX]  -> results/runtime_matrix.json (merged)

Energy per inference = (mean package power over the timed window - idle power) * wall / iterations.
Mac (M5 Pro) numbers; the iPhone needs the same Swift code in an app target.
"""
import argparse, json, os, re, statistics, subprocess, threading, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(HERE, "work", "runtimebench")
M = lambda n: os.path.join(HERE, "work", "models", n)  # noqa: E731
RAILS = ["cpu_power", "gpu_power", "ane_power", "ram_power", "all_power"]

CASES = [
    # (label, runtime, unit, model file)
    ("coreml_fp16_stock", "coreml", "gpu", "EViTSAM_Encoder_l0_fp16_tensor.mlpackage"),
    ("coreml_fp16_stock", "coreml", "ane", "EViTSAM_Encoder_l0_fp16_tensor.mlpackage"),
    ("coreml_fp16_anefix", "coreml", "gpu", "EViTSAM_Encoder_l0_fp16_anefix_tensor.mlpackage"),
    ("coreml_fp16_anefix", "coreml", "ane", "EViTSAM_Encoder_l0_fp16_anefix_tensor.mlpackage"),
    ("coreml_fp32", "coreml", "gpu", "EViTSAM_Encoder_l0_fp32.mlpackage"),
    ("coreai_fp16_stock", "coreai", "gpu", "EViTSAM_Encoder_l0_fp16.aimodel"),
    ("coreai_fp16_stock", "coreai", "ane", "EViTSAM_Encoder_l0_fp16.aimodel"),
    ("coreai_fp16_anefix", "coreai", "gpu", "EViTSAM_Encoder_l0_fp16_anefix.aimodel"),
    ("coreai_fp16_anefix", "coreai", "ane", "EViTSAM_Encoder_l0_fp16_anefix.aimodel"),
    ("coreai_fp32", "coreai", "gpu", "EViTSAM_Encoder_l0_fp32.aimodel"),
    ("coreml_fp16_tf2_s34", "coreml", "ane", "EViTSAM_Encoder_l0_fp16_tf2_s34.mlpackage"),
    ("coreml_fp16_tf2_s3", "coreml", "ane", "EViTSAM_Encoder_l0_fp16_tf2_s3.mlpackage"),
    ("coreai_fp16_tf_s34", "coreai", "ane", "EViTSAM_Encoder_l0_fp16_tf_s34.aimodel"),
    ("coreai_fp16_tf_s34", "coreai", "gpu", "EViTSAM_Encoder_l0_fp16_tf_s34.aimodel"),
    ("coreml_fp16_ef_s34", "coreml", "ane", "EViTSAM_Encoder_l0_fp16_ef_s34.mlpackage"),
    ("coreml_fp16_ef_s34", "coreml", "gpu", "EViTSAM_Encoder_l0_fp16_ef_s34.mlpackage"),
    ("coreai_fp16_ef_s34", "coreai", "ane", "EViTSAM_Encoder_l0_fp16_ef_s34.aimodel"),
    ("coreai_fp16_ef_s34", "coreai", "gpu", "EViTSAM_Encoder_l0_fp16_ef_s34.aimodel"),
    ("mpsgraph_fp16_ef_s34", "mpsgraph", "gpu", "EViTSAM_Encoder_l0_fp16_ef_s34.mlpackage"),
    ("mpsgraph_fp16_stock", "mpsgraph", "gpu", "EViTSAM_Encoder_l0_fp16_tensor.mlpackage"),
    ("mpsgraph_fp16_anefix", "mpsgraph", "gpu", "EViTSAM_Encoder_l0_fp16_anefix_tensor.mlpackage"),
]


class Power:
    def __enter__(self):
        self.s = []
        self.p = subprocess.Popen(["macmon", "pipe", "-i", "100"], stdout=subprocess.PIPE, text=True)
        threading.Thread(target=self._read, daemon=True).start()
        return self

    def _read(self):
        for line in self.p.stdout:
            try:
                d = json.loads(line)
                d["t"] = time.time()
                self.s.append(d)
            except json.JSONDecodeError:
                pass

    def __exit__(self, *a):
        self.p.terminate()
        self.p.wait()

    def mean(self, t0, t1):
        s = [x for x in self.s if t0 <= x["t"] <= t1]
        return {r: statistics.fmean(x.get(r) or 0 for x in s) for r in RAILS} if s else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=8)
    ap.add_argument("--only", default="")
    ap.add_argument("--extra", nargs=4, action="append", default=[], metavar=("LABEL", "RUNTIME", "UNIT", "MODEL"))
    args = ap.parse_args()
    # Pause any distillation run (GPU) for the whole matrix so it does not skew latency or power.
    subprocess.run(["pkill", "-STOP", "-f", "train_distill.py"])
    try:
        run(args)
    finally:
        subprocess.run(["pkill", "-CONT", "-f", "train_distill.py"])


def run(args):
    path = os.path.join(HERE, "results", "runtime_matrix.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    with Power() as p:
        time.sleep(4)
        idle = p.mean(time.time() - 3.5, time.time())
    out["idle_power_w"] = idle
    for label, rt, unit, model in CASES + [tuple(e) for e in args.extra]:
        key = f"{label}/{rt}/{unit}"
        if not re.search(args.only, key):
            continue
        time.sleep(2)  # let clocks settle between cases
        env = dict(os.environ, DURATION=str(args.duration))
        with Power() as p:
            r = subprocess.run([BENCH, rt, unit, M(model)], capture_output=True, text=True, env=env)
            t1 = time.time()
        try:
            d = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:
            print(key, "FAILED", r.stderr[-300:])
            out[key] = {"error": r.stderr[-1000:]}
            continue
        pw = p.mean(t1 - args.duration + 0.5, t1 - 0.2)
        dp = {k: pw[k] - idle.get(k, 0) for k in pw}
        d["power_w"], d["power_delta_w"] = pw, dp
        d["energy_mj"] = {k: v * d["wall_s"] / d["iters"] * 1000 for k, v in dp.items()}
        out[key] = d
        print(f"{key:40s} p50 {d['latency_ms']['p50']:7.2f} ms  p90 {d['latency_ms']['p90']:7.2f}  "
              f"dP {dp.get('all_power', 0):5.2f} W (gpu {dp.get('gpu_power', 0):5.2f} ane {dp.get('ane_power', 0):5.2f})  "
              f"E {d['energy_mj'].get('all_power', 0):6.1f} mJ/inf  load {d['load_ms']:6.0f} ms first {d['first_ms']:7.1f} ms", flush=True)
        json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
