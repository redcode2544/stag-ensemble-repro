#!/usr/bin/env python3
"""Cross-validate STAG on exactly ONE dataset, in its own process.

One process per dataset is deliberate. Running several in a single process
accumulates memory that is never handed back (upstream deepcopies every graph
per fold, and the MPS allocator keeps its pool), so on a 24 GB unified-memory
Mac the second dataset onward dies with "MPS backend out of memory" while the
first succeeds. A fresh process is the only reliable reset.

The fold draw is seeded here, in the runner; nothing in models/ is changed.

Usage: run_one.py <repo> <dataset> <out.json> [seed]
"""
import sys, os, json, time
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
name = sys.argv[2]
out = Path(sys.argv[3])
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 20261107

sys.path.insert(0, str(repo))
os.chdir(repo)

import numpy as np
import pandas as pd
import torch
from models.STAG.cross_validation import cross_validation as cv_STAG
from models.STAG.cross_validation import DEVICE

table = f"datasets/{name}/{name}.csv"
sdir = f"datasets/{name}/structures"
df = pd.read_csv(table, index_col=0)

print(f"{name}: {df.shape[0]} complexes, {df['label'].mean()*100:.1f}% positive, "
      f"device={DEVICE}, seed={seed}", flush=True)

rec = {"complexes": int(df.shape[0]), "positive_rate": float(df["label"].mean()),
       "seed": seed, "device": str(DEVICE)}
np.random.seed(seed)
t0 = time.time()
try:
    a = np.array(cv_STAG(table, sdir, k=5), dtype=float)
    rec.update(
        fold_auc=[float(x) for x in a], mean_auc=float(a.mean()),
        sd_auc=float(a.std(ddof=1)), min_auc=float(a.min()), max_auc=float(a.max()),
        seconds=round(time.time() - t0, 1), status="ok",
    )
    print(f"{name}: mean ROC-AUC {a.mean():.3f} (sd {a.std(ddof=1):.3f}) "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)
except Exception as e:
    rec.update(status="failed", error=f"{type(e).__name__}: {e}",
               seconds=round(time.time() - t0, 1))
    print(f"{name}: FAILED {type(e).__name__}: {e}", flush=True)

if DEVICE.type == "mps":
    peak = getattr(torch.mps, "current_allocated_memory", lambda: None)()
    if peak is not None:
        rec["mps_allocated_bytes_at_end"] = int(peak)

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rec, indent=2))
print(f"wrote {out}", flush=True)
sys.exit(0 if rec["status"] == "ok" else 1)
