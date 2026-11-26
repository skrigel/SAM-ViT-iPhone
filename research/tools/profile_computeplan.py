"""Core ML compute plan: which device each op is dispatched to, and Core ML's estimated relative cost.

Usage: python tools/profile_computeplan.py  -> results/computeplan.json
"""

import glob
import json
import os
import sys
from collections import defaultdict

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evitsam import RESULTS, ROOT  # noqa: E402

UNITS = {"ALL": ct.ComputeUnit.ALL, "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE, "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU}
SKIP = {"const"}


def dev_name(d):
    n = type(d).__name__
    return {"MLCPUComputeDevice": "CPU", "MLGPUComputeDevice": "GPU", "MLNeuralEngineComputeDevice": "ANE"}.get(n, n)


def analyze(path, units):
    compiled = ct.utils.compile_model(path)
    plan = MLComputePlan.load_from_path(compiled, compute_units=units)
    fn = plan.model_structure.program.functions["main"]
    ops = []

    def walk(block):
        for op in block.operations:
            for b in op.blocks:
                walk(b)
            if op.operator_name in SKIP:
                continue
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            cost = plan.get_estimated_cost_for_mlprogram_operation(op)
            ops.append(dict(
                op=op.operator_name,
                out=op.outputs[0].name if op.outputs else "",
                device=dev_name(usage.preferred_compute_device) if usage else "n/a",
                supported=sorted({dev_name(d) for d in usage.supported_compute_devices}) if usage else [],
                cost=cost.weight if cost else 0.0,
            ))

    walk(fn.block)
    by_dev, by_op = defaultdict(lambda: [0, 0.0]), defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    for o in ops:
        by_dev[o["device"]][0] += 1
        by_dev[o["device"]][1] += o["cost"]
        by_op[o["op"]][o["device"]][0] += 1
        by_op[o["op"]][o["device"]][1] += o["cost"]
    not_ane = defaultdict(int)
    for o in ops:
        if "ANE" not in o["supported"]:
            not_ane[o["op"]] += 1
    # Device transitions = places where the graph hops between compute units (each costs a sync + copy).
    transitions = sum(1 for a, b in zip(ops, ops[1:]) if a["device"] != b["device"])
    return dict(
        n_ops=len(ops),
        transitions=transitions,
        by_device={k: dict(n=v[0], cost=v[1]) for k, v in by_dev.items()},
        by_op={k: {d: dict(n=x[0], cost=x[1]) for d, x in v.items()} for k, v in by_op.items()},
        ane_unsupported=dict(not_ane),
        top_ops=sorted(ops, key=lambda o: -o["cost"])[:20],
        sequence=[(o["op"], o["device"], round(o["cost"], 5)) for o in ops],
    )


def main():
    out = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "models", "*_fp16.mlpackage"))):
        name = os.path.basename(path).replace(".mlpackage", "")
        out[name] = {}
        for uname, u in UNITS.items():
            r = analyze(path, u)
            out[name][uname] = r
            print(f"{name:28s} {uname:12s} ops={r['n_ops']:4d} hops={r['transitions']:3d} "
                  + " ".join(f"{d}:{v['n']}({v['cost']*100:.1f}%)" for d, v in r["by_device"].items()))
        print("   not ANE-capable:", out[name]["ALL"]["ane_unsupported"])
    with open(os.path.join(RESULTS, "computeplan.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
