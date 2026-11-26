"""Append a results.tsv row from results/eval/<tag>.json.

  python tools/log_result.py <tag> <keep|discard|crash> "<note>" [--latency MS]
"""
import argparse, datetime, json, os, subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("tag"); ap.add_argument("status"); ap.add_argument("note")
ap.add_argument("--latency", type=float)
ap.add_argument("--experiment")
a = ap.parse_args()
d = json.load(open(os.path.join(HERE, "results", "eval", f"{a.tag}.json")))
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
row = [datetime.date.today().isoformat(), commit, a.experiment or a.tag, d["backend"], d["unit"], d["split"], str(d["n"]),
       f"{d['mIoU']:.2f}", f"{d['ref_mIoU']:.2f}", f"{d['delta']:+.2f}", f"{d['cos_mean']:.4f}",
       "" if a.latency is None else f"{a.latency:.2f}", a.status, a.note]
open(os.path.join(HERE, "results.tsv"), "a").write("\t".join(row) + "\n")
print("\t".join(row))
