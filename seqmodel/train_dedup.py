#!/usr/bin/env python3
"""Apply the filter the STAG paper says it applied, to the TRAINING folds only.

Section III of the paper: the datasets "were pruned so that there were no
decoys with kernel similarity greater than 0.90 and no ground truth samples
with kernel similarity greater than 0.95". The released peptide-specific files
do not satisfy that: every one of them contains CDR3 alpha+beta pairs that are
IDENTICAL -- similarity 1.0, above both thresholds -- and carry opposite
labels. Two rows with the same input and different labels cannot both be right,
and no function can satisfy both, so they are pure noise in training.

Their similarity kernel is not published with the data, so this applies only
the part that needs no kernel: exact duplicates.
  * a group of identical (CDR3a, CDR3b) carrying BOTH labels -> dropped whole
  * a group of identical pairs carrying one label -> collapsed to one row

The test fold is never touched, so the benchmark it is scored against is still
the published one. SeqNoDedup is the identical model with the filter off, so
the difference is the filter and nothing else. Representation is fixed to
cdr3+full ("text_both") in both, to keep the two runs comparable fit for fit.
"""
import argparse, csv, glob, json, os, pathlib, sys
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_seq import load_dataset, auc, SEQ                 # noqa: E402

HERE  = pathlib.Path(__file__).resolve().parent.parent
OUT   = HERE / "runs" / "results" / "preds_dedup"
FIELD = "text_both"
CGRID = [0.1, 0.5, 1.0, 4.0, 16.0]

def prune(rows):
    """Paper's own rule, exact-duplicate part. -> kept rows, stats."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["a"], r["b"])].append(r)
    keep, conflict, collapsed = [], 0, 0
    for g in groups.values():
        labels = {r["y"] for r in g}
        if len(labels) > 1:
            conflict += len(g); continue
        if len(g) > 1:
            collapsed += len(g) - 1
        keep.append(g[0])
    return keep, dict(dropped_conflicting=conflict, collapsed_duplicates=collapsed)

def fit_predict(tr, te, seed):
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=2,
                          sublinear_tf=True)
    X = vec.fit_transform([r[FIELD] for r in tr])
    y = np.array([r["y"] for r in tr])
    gs = GridSearchCV(LogisticRegression(max_iter=4000, solver="liblinear"),
                      {"C": CGRID}, scoring="roc_auc",
                      cv=StratifiedKFold(4, shuffle=True, random_state=seed),
                      n_jobs=1).fit(X, y)
    return gs.best_estimator_.decision_function(
        vec.transform([r[FIELD] for r in te]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--inits", type=int, default=2)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    summary = {}
    for ds in (args.datasets or sorted(os.path.basename(p)[:-4]
                                       for p in glob.glob(str(SEQ / "*.csv")))):
        rows, _ = load_dataset(ds)
        folds = sorted({r["fold"] for r in rows})
        per = defaultdict(lambda: defaultdict(list)); stats = None
        for init in range(1, args.inits + 1):
            for k in folds:
                tr = [r for r in rows if r["fold"] != k]
                te = [r for r in rows if r["fold"] == k]
                yte = np.array([r["y"] for r in te])
                kept, stats = prune(tr)
                preds = {"SeqDedup": fit_predict(kept, te, init),
                         "SeqNoDedup": fit_predict(tr, te, init)}
                for m, v in preds.items():
                    per[m][init].append(auc(yte, v))
                    if args.write:
                        d = OUT / f"init{init}"; d.mkdir(parents=True, exist_ok=True)
                        with open(d / f"{m}__{ds}_fold{k}.csv", "w", newline="") as fh:
                            w = csv.writer(fh)
                            w.writerow(["index", "label", "pred", "fold"])
                            for r, x in zip(te, v):
                                w.writerow([r["index"], r["y"], float(x), k])
        line = []
        for m in ("SeqNoDedup", "SeqDedup"):
            means = [float(np.mean(per[m][i])) for i in sorted(per[m])]
            line.append(f"{m} {np.mean(means):.3f}")
            summary.setdefault(ds, {})[m] = dict(mean=float(np.mean(means)),
                                                 spread=float(max(means) - min(means)),
                                                 per_init=means)
        summary[ds]["_prune"] = stats
        d = summary[ds]["SeqDedup"]["mean"] - summary[ds]["SeqNoDedup"]["mean"]
        print(f"{ds:11s} rows {len(rows):5d} | dropped-conflicting "
              f"{stats['dropped_conflicting']:4d} collapsed {stats['collapsed_duplicates']:4d}"
              f" | " + "  ".join(line) + f"  delta {d:+.3f}", flush=True)
    if args.write:
        json.dump(summary, open(HERE / "runs" / "results" / f"dedup_summary{args.tag}.json", "w"),
                  indent=1)
        print(f"\nwrote {OUT} and runs/results/dedup_summary{args.tag}.json")

if __name__ == "__main__":
    main()
