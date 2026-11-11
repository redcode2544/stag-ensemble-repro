#!/usr/bin/env python3
"""Does each peptide's POSITIVE set carry a repertoire signature at all?

Two questions, both answered with numbers rather than argument.

1. Pairwise separability of positive sets.
   Take the positives of peptide A as class 1 and the positives of peptide B as
   class 0, drop every TCR that appears in both, and cross-validate. If A's
   positives are real binders they must look different from B's -- especially
   when A and B are presented by different HLA alleles, because the V-gene
   usage of a genuinely restricted repertoire differs. An AUC near 0.5 against
   every partner means that peptide's "positives" are a random draw from the
   shared donor repertoire, i.e. the label carries no information. This never
   touches the negatives, so it is independent of how they were constructed.

2. The flip rate needed to explain the measured AUC.
   For a perfect scorer and a symmetric label-flip rate rho, the observed AUC
   is a known function of rho and the class prevalence. Invert it: how wrong
   would the labels have to be for a PERFECT model to score what we measure?
   That is an upper bound on the noise, and the honest alternative reading --
   "there is no signal" -- is the other end of the same statement.

Reads the folds and sequences already on disk. Writes
runs/results/label_probe.json.
"""
import csv, glob, json, os, pathlib, sys
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_seq import load_dataset, auc, SEQ                 # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
OUT  = HERE / "runs" / "results" / "label_probe.json"

# HLA restriction of each dataset's epitope, from the papers the datasets cite.
HLA = {
    "ELAGIGILTV": "A*02:01", "NLVPMVATV": "A*02:01",
    "GILGFVFTL":  "A*02:01", "GLCTLVAML": "A*02:01",
    "RAKFKQLL":   "B*08:01", "KLGGALQAK": "A*03:01",
    "AVFDRKSDAK": "A*11:01", "IVTDFSVIK": "A*11:01",
}

def xval_auc(texts, y, seed=1, folds=5):
    """Pooled out-of-fold AUC of an n-gram logistic regression."""
    y = np.asarray(y)
    if min(int((y == 1).sum()), int((y == 0).sum())) < folds:
        return float("nan")
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in skf.split(np.zeros(len(y)), y):
        vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=2,
                              sublinear_tf=True)
        Xtr = vec.fit_transform([texts[i] for i in tr])
        Xte = vec.transform([texts[i] for i in te])
        gs = GridSearchCV(LogisticRegression(max_iter=4000, solver="liblinear"),
                          {"C": [0.1, 1.0, 4.0]}, scoring="roc_auc",
                          cv=StratifiedKFold(4, shuffle=True, random_state=seed),
                          n_jobs=1).fit(Xtr, y[tr])
        oof[te] = gs.best_estimator_.decision_function(Xte)
    return float(auc(y, oof))

def observed_auc(rho, prev):
    """AUC a PERFECT scorer shows when labels flip at rate rho both ways."""
    p, q = prev, 1.0 - prev
    a = (1 - rho) * p / ((1 - rho) * p + rho * q)      # P(truly +| labelled +)
    b = (1 - rho) * q / ((1 - rho) * q + rho * p)      # P(truly -| labelled -)
    return a * b + 0.5 * (a * (1 - b) + (1 - a) * b)

def flip_rate_for(target, prev):
    """Invert observed_auc: the rho that drags a perfect model to `target`."""
    lo, hi = 0.0, 0.5
    if observed_auc(hi, prev) > target:                 # even 50% cannot explain it
        return float("nan")
    for _ in range(80):
        mid = (lo + hi) / 2
        if observed_auc(mid, prev) > target: lo = mid
        else: hi = mid
    return (lo + hi) / 2

def main():
    datasets = sorted(os.path.basename(p)[:-4] for p in glob.glob(str(SEQ / "*.csv")))
    data, prev = {}, {}
    for ds in datasets:
        rows, _ = load_dataset(ds)
        data[ds] = rows
        prev[ds] = sum(r["y"] for r in rows) / len(rows)

    # ---- 1. pairwise positives-vs-positives -----------------------------
    pos = {ds: [r for r in data[ds] if r["y"] == 1] for ds in datasets}
    pair = {}
    for i, A in enumerate(datasets):
        for B in datasets[i + 1:]:
            ida = {(r["a"], r["b"]) for r in pos[A]}
            idb = {(r["a"], r["b"]) for r in pos[B]}
            both = ida & idb
            ra = [r for r in pos[A] if (r["a"], r["b"]) not in both]
            rb = [r for r in pos[B] if (r["a"], r["b"]) not in both]
            texts = [r["text_cdr3"] for r in ra + rb]
            y = [1] * len(ra) + [0] * len(rb)
            a = xval_auc(texts, y)
            pair[f"{A}|{B}"] = dict(auc=a, na=len(ra), nb=len(rb),
                                    shared=len(both),
                                    same_hla=HLA[A] == HLA[B])
            print(f"  {A:11s} vs {B:11s} n={len(ra):4d}/{len(rb):4d} "
                  f"shared={len(both):3d} {'same' if HLA[A]==HLA[B] else 'diff'}-HLA "
                  f"AUC={a:.3f}", flush=True)

    # ---- 2. one informativeness number per dataset ----------------------
    one = {}
    for A in datasets:
        ida = {(r["a"], r["b"]) for r in pos[A]}
        other = [r for B in datasets if B != A for r in pos[B]
                 if (r["a"], r["b"]) not in ida]
        texts = [r["text_cdr3"] for r in pos[A]] + [r["text_cdr3"] for r in other]
        y = [1] * len(pos[A]) + [0] * len(other)
        a = xval_auc(texts, y)
        one[A] = dict(auc=a, n_pos=len(pos[A]), n_other=len(other))
        print(f"  {A:11s} positives vs all other peptides' positives "
              f"({len(pos[A])} vs {len(other)}): AUC={a:.3f}", flush=True)

    # ---- 3. flip rate that would explain the measured fusion AUC --------
    fa = json.load(open(HERE / "runs" / "results" / "fusion_all.json"))
    ceil = {}
    for ds in datasets:
        got = fa[ds]["fusion"]["auto"]["auc"]
        rho = flip_rate_for(got, prev[ds])
        ceil[ds] = dict(auc=got, prevalence=prev[ds], flip_rate=rho,
                        n=len(data[ds]))
        print(f"  {ds:11s} auto={got:.3f} prev={prev[ds]:.3f} -> "
              f"rho={rho:.3f}", flush=True)

    json.dump(dict(pairwise=pair, one_vs_rest=one, flip=ceil, hla=HLA),
              open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")

if __name__ == "__main__":
    main()
