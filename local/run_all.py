#!/usr/bin/env python3
"""Cross-validate STAG over several datasets, one subprocess per dataset, and
merge the per-dataset JSON into one results file as each finishes.

Each dataset runs in its own process on purpose — see run_one.py for why.

Usage: run_all.py <repo> [out.json] [dataset ...]
"""
import sys, json, subprocess
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]) if len(sys.argv) > 2 else repo / "runs/stag_results.json"
SEED = 20261107

# smallest first, so partial results are useful if the batch is cut short
DEFAULT = ["IVTDFSVIK", "GLCTLVAML", "AVFDRKSDAK", "NLVPMVATV",
           "KLGGALQAK", "ELAGIGILTV", "RAKFKQLL", "GILGFVFTL"]
datasets = sys.argv[3:] or DEFAULT

per = repo / "runs/results"
per.mkdir(parents=True, exist_ok=True)
merged = json.loads(out.read_text())["datasets"] if out.is_file() else {}


def write_merged() -> None:
    out.write_text(json.dumps({
        "method": "STAG",
        "seed": SEED,
        "k_folds": 5,
        "upstream_commit": "06c30a57",
        "metric_selected_on": "test fold (upstream protocol — optimistically biased)",
        "isolation": "one subprocess per dataset",
        "datasets": merged,
    }, indent=2))


for name in datasets:
    target = per / f"{name}.json"
    print(f"\n{'='*64}\nlaunching {name}", flush=True)
    subprocess.run([sys.executable, str(repo / "local/run_one.py"),
                    str(repo), name, str(target), str(SEED)], check=False)
    if target.is_file():
        merged[name] = json.loads(target.read_text())
    else:
        merged[name] = {"status": "failed",
                        "error": "subprocess wrote no result file"}
    write_merged()
    r = merged[name]
    print(f"{name}: {r['status']}"
          + (f" mean {r['mean_auc']:.3f}" if r.get("status") == "ok" else ""),
          flush=True)

write_merged()
ok = [n for n, r in merged.items() if r.get("status") == "ok"]
print(f"\ndone: {len(ok)}/{len(merged)} ok -> {out}", flush=True)
