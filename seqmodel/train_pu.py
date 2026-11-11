#!/usr/bin/env python3
"""Two members built from what the diagnostics actually found.

SeqPU -- bagging positive-unlabelled learning (Mordelet & Vert 2014).
    The negative-provenance analysis showed that every dataset scores worst
    against the negatives of unknown provenance, i.e. cells the dextramer
    failed to stain, which is where missed binders sit. Standard training
    treats each of those as a confident negative. Bagging PU instead trains
    many classifiers, each on all the positives against a small random draw of
    the negatives, and averages. A contaminated negative then influences only
    the few bags that drew it, so its pull on the boundary is diluted rather
    than taken at face value. This is the textbook remedy for the defect that
    was measured, not a guess.

SeqAvg3 -- average the three sequence representations instead of choosing one.
    SeqLR picks CDR3-only / full-chain / both by inner CV on four folds, which
    is a noisy choice; averaging the three rank-normalised scores spends that
    variance instead of betting on it. Same idea as the committee rule in
    fuse_all.py, one level down.

Same folds as every other member, read back from the structure runs.
"""
import argparse, csv, glob, json, os, pathlib, sys
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_seq import load_dataset, auc, SEQ, FIELDS         # noqa: E402

HERE  = pathlib.Path(__file__).resolve().parent.parent
OUT   = HERE / "runs" / "results" / "preds_pu"
BAGS  = 25
CGRID = [0.1, 0.5, 1.0, 4.0, 16.0]

def ranks(v):
    v = np.asarray(v, float)
    return v.argsort().argsort().astype(float) / max(len(v) - 1, 1)

def best_C(texts, y, seed):
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=2,
                          sublinear_tf=True)
    X = vec.fit_transform(texts)
    gs = GridSearchCV(LogisticRegression(max_iter=4000, solver="liblinear"),
                      {"C": CGRID}, scoring="roc_auc",
                      cv=StratifiedKFold(4, shuffle=True, random_state=seed),
                      n_jobs=1).fit(X, np.asarray(y))
    return gs.best_params_["C"]

def fit_predict(tr_t, tr_y, te_t, C):
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=2,
                          sublinear_tf=True)
    X = vec.fit_transform(tr_t)
    lr = LogisticRegression(max_iter=4000, solver="liblinear", C=C).fit(X, tr_y)
    return lr.decision_function(vec.transform(te_t))

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
        per = defaultdict(lambda: defaultdict(list))
        for init in range(1, args.inits + 1):
            seed = init
            rng = np.random.default_rng(1000 + seed)
            for k in folds:
                tr = [r for r in rows if r["fold"] != k]
                te = [r for r in rows if r["fold"] == k]
                yte = np.array([r["y"] for r in te])

                # ---- SeqPU
                field = "text_both"
                pos = [r for r in tr if r["y"] == 1]
                neg = [r for r in tr if r["y"] == 0]
                C = best_C([r[field] for r in tr], [r["y"] for r in tr], seed)
                acc = np.zeros(len(te))
                for _ in range(BAGS):
                    take = rng.choice(len(neg), size=min(len(pos), len(neg)),
                                      replace=False)
                    bag = pos + [neg[i] for i in take]
                    by = np.array([1] * len(pos) + [0] * len(take))
                    acc += ranks(fit_predict([r[field] for r in bag], by,
                                             [r[field] for r in te], C))
                pu = acc / BAGS

                # ---- SeqAvg3
                ytr = np.array([r["y"] for r in tr])
                a3 = np.zeros(len(te))
                for f in FIELDS:
                    Cf = best_C([r[f] for r in tr], ytr, seed)
                    a3 += ranks(fit_predict([r[f] for r in tr], ytr,
                                            [r[f] for r in te], Cf))
                a3 /= len(FIELDS)

                for m, v in (("SeqPU", pu), ("SeqAvg3", a3)):
                    per[m][init].append(auc(yte, v))
                    if args.write:
                        d = OUT / f"init{init}"; d.mkdir(parents=True, exist_ok=True)
                        with open(d / f"{m}__{ds}_fold{k}.csv", "w", newline="") as fh:
                            w = csv.writer(fh)
                            w.writerow(["index", "label", "pred", "fold"])
                            for r, x in zip(te, v):
                                w.writerow([r["index"], r["y"], float(x), k])
        line = []
        for m in ("SeqPU", "SeqAvg3"):
            means = [float(np.mean(per[m][i])) for i in sorted(per[m])]
            line.append(f"{m} {np.mean(means):.3f}+-{max(means)-min(means):.3f}")
            summary.setdefault(ds, {})[m] = dict(mean=float(np.mean(means)),
                                                 spread=float(max(means) - min(means)),
                                                 per_init=means)
        print(f"{ds:11s} bags={BAGS} | " + "  ".join(line), flush=True)
    if args.write:
        json.dump(summary, open(HERE / "runs" / "results" / f"pu_summary{args.tag}.json", "w"),
                  indent=1)
        print(f"\nwrote {OUT} and runs/results/pu_summary{args.tag}.json")

if __name__ == "__main__":
    main()
