"""Print one line per results/eval/*.json (sorted by modification time)."""
import glob, json, os, sys
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for f in sorted(glob.glob(os.path.join(here, "results", "eval", "*.json")), key=os.path.getmtime):
    d = json.load(open(f))
    print("%-34s %-6s %-5s n=%3d mIoU %6.2f ref %6.2f delta %+6.2f cos %.4f (min %.4f) enc %5.1f ms"
          % (d["tag"], d["backend"], d["unit"], d["n"], d["mIoU"], d["ref_mIoU"], d["delta"], d["cos_mean"], d["cos_min"], d["enc_ms_median"] or 0))
