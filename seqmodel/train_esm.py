#!/usr/bin/env python3
"""Fifth stream: a protein language model (ESM-2) over the TCR V-domains.

Same folds as everything else, read back from the structure runs. The
embeddings are computed once by embed_esm.py and cached, so this file only
fits the classifier. Two members, because a language-model embedding is a
dense continuous space and the right classifier for it is not obvious:

  SeqESM     logistic regression on the standardised embedding
  SeqESMNN   cosine nearest neighbours in embedding space

Nothing about the test fold enters the scaler or the model fit.
"""
import argparse, csv, glob, json, os, pathlib, sys
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_seq import load_dataset, auc, SEQ                 # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
OUT  = HERE / "runs" / "results" / "preds_esm"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esm2_t12_35M_UR50D")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--inits", type=int, default=2)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    z = np.load(HERE / "runs" / "results" / "esm" / f"{args.model}.npz",
                allow_pickle=True)
    idx = {s: i for i, s in enumerate(z["seqs"].tolist())}
    V = z["vecs"]
    print(f"embeddings {V.shape} for {len(idx)} chains", flush=True)

    summary = {}
    for ds in (args.datasets or sorted(os.path.basename(p)[:-4]
                                       for p in glob.glob(str(SEQ / "*.csv")))):
        rows, _ = load_dataset(ds)
        keep = [r for r in rows if r["fa"] in idx and r["fb"] in idx]
        X = np.hstack([V[[idx[r["fa"]] for r in keep]],
                       V[[idx[r["fb"]] for r in keep]]])
        y = np.array([r["y"] for r in keep])
        fold = np.array([r["fold"] for r in keep])
        per = defaultdict(lambda: defaultdict(list))
        for init in range(1, args.inits + 1):
            seed = init
            inner = StratifiedKFold(4, shuffle=True, random_state=seed)
            for k in sorted(set(fold.tolist())):
                tr, te = fold != k, fold == k
                gs = GridSearchCV(
                    make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=3000)),
                    {"logisticregression__C": [0.003, 0.03, 0.3, 3.0]},
                    scoring="roc_auc", cv=inner, n_jobs=1).fit(X[tr], y[tr])
                s = gs.best_estimator_.decision_function(X[te])
                gk = GridSearchCV(KNeighborsClassifier(metric="cosine",
                                                       weights="distance"),
                                  {"n_neighbors": [5, 11, 21, 41]},
                                  scoring="roc_auc", cv=inner, n_jobs=1).fit(X[tr], y[tr])
                sk = gk.best_estimator_.predict_proba(X[te])[:, 1]
                for m, v in (("SeqESM", s), ("SeqESMNN", sk)):
                    per[m][init].append(auc(y[te], v))
                    if args.write:
                        d = OUT / f"init{init}"; d.mkdir(parents=True, exist_ok=True)
                        with open(d / f"{m}__{ds}_fold{k}.csv", "w", newline="") as fh:
                            w = csv.writer(fh)
                            w.writerow(["index", "label", "pred", "fold"])
                            for r, x in zip([q for q, t in zip(keep, te) if t], v):
                                w.writerow([r["index"], r["y"], float(x), k])
        line = []
        for m in ("SeqESM", "SeqESMNN"):
            means = [float(np.mean(per[m][i])) for i in sorted(per[m])]
            line.append(f"{m} {np.mean(means):.3f}+-{max(means)-min(means):.3f}")
            summary.setdefault(ds, {})[m] = dict(mean=float(np.mean(means)),
                                                 spread=float(max(means) - min(means)),
                                                 per_init=means)
        print(f"{ds:11s} n={len(keep):5d} dim={X.shape[1]} | " + "  ".join(line), flush=True)
    if args.write:
        json.dump(summary, open(HERE / "runs" / "results" / f"esm_summary{args.tag}.json", "w"),
                  indent=1)
        print(f"\nwrote {OUT} and runs/results/esm_summary{args.tag}.json")

if __name__ == "__main__":
    main()
