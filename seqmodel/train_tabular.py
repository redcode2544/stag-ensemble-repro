#!/usr/bin/env python3
"""Train on a tabular feature table and dump predictions on the EXISTING folds.

Used for the richer contact descriptors (runs/../seq/contacts/*.csv), which keep
the geometry the shipped Contacts baseline discards. Folds are read back from the
structure runs so every number stays paired with them.

Members: ConGBM (gradient-boosted trees), ConLR (regularised linear), both with
their hyperparameters chosen by inner CV on the training folds only.
"""
import csv, glob, json, os, pathlib, sys, argparse
from collections import defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = pathlib.Path(__file__).resolve().parent.parent
REF  = HERE / "runs" / "results" / "preds_seed" / "init1"

def load_folds(ds):
    fold = {}
    for f in sorted(glob.glob(str(REF / f"STAG__{ds}_fold*.csv"))):
        k = int(os.path.basename(f)[:-4].rsplit("_fold", 1)[1])
        for r in csv.DictReader(open(f)):
            fold[str(int(float(r["index"])))] = k
    return fold

def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if not pos or not neg: return float("nan")
    order = np.argsort(s, kind="mergesort"); ss = s[order]
    ranks = np.empty(len(s)); i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]: j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return (ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="dir of <DS>.csv feature tables")
    ap.add_argument("--prefix", default="Con", help="member name prefix")
    ap.add_argument("--inits", type=int, default=3)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    FEAT = pathlib.Path(args.features)
    OUT  = HERE / "runs" / "results" / "preds_con"
    all_ds = sorted(os.path.basename(p)[:-4] for p in glob.glob(str(FEAT / "*.csv")))
    summary = {}
    for ds in (args.datasets or all_ds):
        rows = list(csv.DictReader(open(FEAT / f"{ds}.csv")))
        names = [c for c in rows[0] if c not in ("index", "label")]
        fold = load_folds(ds)
        keep = [r for r in rows if str(int(float(r["index"]))) in fold]
        X = np.array([[float(r[c]) for c in names] for r in keep])
        y = np.array([int(float(r["label"])) for r in keep])
        fl = np.array([fold[str(int(float(r["index"])))] for r in keep])
        idx = [str(int(float(r["index"]))) for r in keep]
        folds = sorted(set(fl.tolist()))
        per = defaultdict(lambda: defaultdict(list))
        for init in range(1, args.inits + 1):
            inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=init)
            for k in folds:
                tr, te = fl != k, fl == k
                gbm = GridSearchCV(
                    HistGradientBoostingClassifier(early_stopping=True,
                                                   validation_fraction=0.2,
                                                   random_state=init),
                    {"max_iter": [300], "learning_rate": [0.05, 0.1],
                     "max_leaf_nodes": [7, 15, 31], "l2_regularization": [0.0, 1.0]},
                    scoring="roc_auc", cv=inner, n_jobs=-1).fit(X[tr], y[tr])
                lr = GridSearchCV(
                    make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=4000)),
                    {"logisticregression__C": [0.03, 0.1, 0.3, 1.0, 3.0]},
                    scoring="roc_auc", cv=inner, n_jobs=-1).fit(X[tr], y[tr])
                preds = {f"{args.prefix}GBM": gbm.predict_proba(X[te])[:, 1],
                         f"{args.prefix}LR":  lr.decision_function(X[te])}
                for m, s in preds.items():
                    per[m][init].append(auc(y[te], s))
                    if args.write:
                        d = OUT / f"init{init}"; d.mkdir(parents=True, exist_ok=True)
                        with open(d / f"{m}__{ds}_fold{k}.csv", "w", newline="") as fh:
                            w = csv.writer(fh); w.writerow(["index", "label", "pred", "fold"])
                            for i2, yy, v in zip(np.array(idx)[te], y[te], s):
                                w.writerow([i2, int(yy), float(v), k])
        line = []
        for m in sorted(per):
            means = [float(np.mean(per[m][i])) for i in sorted(per[m])]
            line.append(f"{m} {np.mean(means):.3f}±{max(means)-min(means):.3f}")
            summary.setdefault(ds, {})[m] = dict(mean=float(np.mean(means)),
                                                 spread=float(max(means) - min(means)),
                                                 per_init=means)
        print(f"{ds:12s} n={len(keep):5d} nfeat={len(names):3d} | " + "  ".join(line), flush=True)
    if args.write:
        json.dump(summary, open(HERE / "runs" / "results" / "con_summary.json", "w"), indent=1)
        print(f"\nwrote {OUT} and runs/results/con_summary.json")

if __name__ == "__main__":
    main()
