#!/usr/bin/env python3
"""Confident-learning member: prune suspected-mislabelled TRAINING rows.

The negative provenance analysis showed where these datasets hurt: for every
one of the eight, the AUC against negatives of unknown provenance -- cells the
dextramer simply failed to stain -- is lower than against negatives known to
bind another peptide. That is the signature of dropout, i.e. real binders sitting
in the negative pool. Those rows teach the model the wrong boundary.

So: estimate each training row's label agreement from out-of-fold predictions
computed inside the training folds, drop the most contradictory fraction, and
refit. The fraction is chosen by inner CV, and it is chosen against UNPRUNED
held-out rows -- pruning the evaluation set as well would make any fraction
look good by construction.

Nothing is ever pruned from the test fold. Noisy TEST labels still cap the
achievable AUC; this can only move the model closer to that cap, never past it.
Nested exactly: the pruning decision for an inner training part uses
out-of-fold scores computed within that part alone, so the inner test rows'
labels never influence which rows are dropped.
"""
import argparse, csv, glob, json, os, pathlib, sys
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_seq import load_dataset, auc, SEQ                 # noqa: E402

HERE  = pathlib.Path(__file__).resolve().parent.parent
OUT   = HERE / "runs" / "results" / "preds_clean"
FRACS = (0.0, 0.05, 0.10, 0.20)
C_FIX = 1.0

def fit(texts, y, C=C_FIX):
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=2,
                          sublinear_tf=True)
    X = vec.fit_transform(texts)
    lr = LogisticRegression(max_iter=4000, solver="liblinear", C=C).fit(X, y)
    return vec, lr

def oof_scores(texts, y, seed, k=4):
    """Out-of-fold score for every row, using only the other rows of this set."""
    y = np.asarray(y); s = np.zeros(len(y))
    skf = StratifiedKFold(min(k, int(min((y == 1).sum(), (y == 0).sum()))),
                          shuffle=True, random_state=seed)
    for a, b in skf.split(np.zeros(len(y)), y):
        vec, lr = fit([texts[i] for i in a], y[a])
        s[b] = lr.decision_function(vec.transform([texts[i] for i in b]))
    return s

def prune(idx, texts, y, scores, frac):
    """Drop the `frac` most label-contradicting rows, proportional per class."""
    if frac <= 0: return list(idx)
    y = np.asarray(y)
    rank = scores.argsort().argsort().astype(float)
    rank = rank / max(len(rank) - 1, 1)                 # 0 = lowest score
    drop = set()
    for cls, key in ((1, lambda i: rank[i]), (0, lambda i: -rank[i])):
        members = [i for i in range(len(y)) if y[i] == cls]
        n = int(round(frac * len(members)))
        drop |= set(sorted(members, key=key)[:n])
    return [i for i in range(len(y)) if i not in drop]

def choose_frac(tr, seed):
    """Inner CV. Prune only the inner TRAINING part; score unpruned held-out."""
    ytr = np.array([r["y"] for r in tr]); ttr = [r["text_cdr3"] for r in tr]
    inner = StratifiedKFold(4, shuffle=True, random_state=seed)
    got = {f: [] for f in FRACS}
    for a, b in inner.split(np.zeros(len(ytr)), ytr):
        ta = [ttr[i] for i in a]; ya = ytr[a]
        if min(int((ya == 1).sum()), int((ya == 0).sum())) < 4: continue
        sa = oof_scores(ta, ya, seed)
        for f in FRACS:
            keep = prune(range(len(ya)), ta, ya, sa, f)
            if len(set(ya[keep].tolist())) < 2: continue
            vec, lr = fit([ta[i] for i in keep], ya[keep])
            got[f].append(auc(ytr[b], lr.decision_function(
                vec.transform([ttr[i] for i in b]))))
    best = max((f for f in FRACS if got[f]),
               key=lambda f: float(np.mean(got[f])), default=0.0)
    return best, (float(np.mean(got[best])) if got[best] else float("nan"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--inits", type=int, default=2)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    all_ds = sorted(os.path.basename(p)[:-4] for p in glob.glob(str(SEQ / "*.csv")))
    summary = {}
    for ds in (args.datasets or all_ds):
        rows, _ = load_dataset(ds)
        folds = sorted({r["fold"] for r in rows})
        per = defaultdict(list); chosen = defaultdict(int); dropped = []
        for init in range(1, args.inits + 1):
            seed = init; aucs = []
            for k in folds:
                tr = [r for r in rows if r["fold"] != k]
                te = [r for r in rows if r["fold"] == k]
                frac, icv = choose_frac(tr, seed)
                chosen[frac] += 1
                ttr = [r["text_cdr3"] for r in tr]; ytr = np.array([r["y"] for r in tr])
                s = oof_scores(ttr, ytr, seed)
                keep = prune(range(len(ytr)), ttr, ytr, s, frac)
                dropped.append(len(ytr) - len(keep))
                vec, lr = fit([ttr[i] for i in keep], ytr[keep])
                v = lr.decision_function(vec.transform([r["text_cdr3"] for r in te]))
                aucs.append(auc(np.array([r["y"] for r in te]), v))
                if args.write:
                    d = OUT / f"init{init}"; d.mkdir(parents=True, exist_ok=True)
                    with open(d / f"SeqClean__{ds}_fold{k}.csv", "w", newline="") as fh:
                        w = csv.writer(fh); w.writerow(["index", "label", "pred", "fold"])
                        for r, x in zip(te, v): w.writerow([r["index"], r["y"], float(x), k])
            per["SeqClean"].append(float(np.mean(aucs)))
        means = per["SeqClean"]
        summary[ds] = dict(mean=float(np.mean(means)),
                           spread=float(max(means) - min(means)), per_init=means,
                           fracs={str(k): v for k, v in sorted(chosen.items())},
                           mean_rows_dropped=float(np.mean(dropped)))
        print(f"{ds:11s} SeqClean {np.mean(means):.3f}+-{max(means)-min(means):.3f} | "
              f"frac=" + ",".join(f"{k}x{v}" for k, v in sorted(chosen.items()))
              + f" | dropped {np.mean(dropped):.0f} rows/fold", flush=True)
    if args.write:
        json.dump(summary, open(HERE / "runs" / "results" / f"clean_summary{args.tag}.json", "w"),
                  indent=1)
        print(f"\nwrote {OUT} and runs/results/clean_summary{args.tag}.json")

if __name__ == "__main__":
    main()
