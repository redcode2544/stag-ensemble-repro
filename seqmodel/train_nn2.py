#!/usr/bin/env python3
"""Nearest published binder: no training at all, just a lookup.

For an epitope with thousands of curated TCRs this is what the field actually
does -- TCRdist / GLIPH annotate a new receptor by how close it sits to known
binders. It is a different kind of member from everything else in the pool:
no fit, no parameters, and the only thing it knows comes from outside the
dataset.

score(x) = mean top-k cosine similarity to the curated binders of THIS epitope
         - mean top-k cosine similarity to the curated binders of OTHER epitopes

Same leak guard as train_ext.py: any curated row whose CDR3-beta appears
anywhere in the target dataset is removed first, so a test TCR can never match
itself or a clone of itself.
"""
import argparse, csv, glob, json, os, pathlib, random, sys
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_seq import load_dataset, auc, SEQ                 # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
EXT  = HERE / "ext"
OUT  = HERE / "runs" / "results" / "preds_nn2"
TENX = "10xgenomics.com"
BG_CAP = 20000

def text(a, b): return f"a{a}|b{b}"

def topk_mean(S, k):
    """Row-wise mean of the k largest values of a dense similarity block."""
    k = min(k, S.shape[1])
    if k == 0: return np.zeros(S.shape[0])
    part = np.partition(S, -k, axis=1)[:, -k:]
    return part.mean(axis=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--guard", choices=("beta", "pair"), default="beta")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    all_ds = sorted(os.path.basename(p)[:-4] for p in glob.glob(str(SEQ / "*.csv")))
    summary = {}
    for ds in (args.datasets or all_ds):
        rows, _ = load_dataset(ds)
        own_pair = {(r["a"], r["b"]) for r in rows}; own_b = {r["b"] for r in rows}
        def blocked(a, b):
            return (a, b) in own_pair or (args.guard == "beta" and b in own_b)

        pos = [r for r in csv.DictReader(open(EXT / f"curated_{ds}_indep.csv"))
               if TENX not in r["ref"] and not blocked(r["a"], r["b"])]
        if len(pos) < 20:
            print(f"{ds:11s} only {len(pos)} curated binders after the guard — skipped",
                  flush=True)
            summary[ds] = dict(skipped=True, curated=len(pos)); continue
        pos_b = {r["b"] for r in pos} | own_b
        bg = [r for r in csv.DictReader(open(EXT / "vdjdb_background.csv"))
              if not blocked(r["a"], r["b"]) and r["b"] not in pos_b]
        if len(bg) > BG_CAP: bg = random.Random(7).sample(bg, BG_CAP)

        vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=1,
                              sublinear_tf=True)
        vec.fit([text(r["a"], r["b"]) for r in pos + bg])
        Ppos = normalize(vec.transform([text(r["a"], r["b"]) for r in pos]))
        Pbg  = normalize(vec.transform([text(r["a"], r["b"]) for r in bg]))

        folds = sorted({r["fold"] for r in rows}); aucs = []
        for k in folds:
            te = [r for r in rows if r["fold"] == k]
            T = normalize(vec.transform([r["text_cdr3"] for r in te]))
            s = topk_mean((T @ Ppos.T).toarray(), args.k) - \
                topk_mean((T @ Pbg.T).toarray(), args.k)
            aucs.append(auc(np.array([r["y"] for r in te]), s))
            if args.write:
                d = OUT / "init1"; d.mkdir(parents=True, exist_ok=True)
                with open(d / f"SeqNN2__{ds}_fold{k}.csv", "w", newline="") as fh:
                    w = csv.writer(fh); w.writerow(["index", "label", "pred", "fold"])
                    for r, v in zip(te, s): w.writerow([r["index"], r["y"], float(v), k])
        summary[ds] = dict(mean=float(np.mean(aucs)), per_fold=aucs,
                           curated=len(pos), background=len(bg), k=args.k)
        print(f"{ds:11s} SeqNN2 {np.mean(aucs):.3f}  (curated {len(pos):6d} vs bg {len(bg):6d})",
              flush=True)
    if args.write:
        json.dump(summary, open(HERE / "runs" / "results" / f"nn2_summary{args.tag}.json", "w"),
                  indent=1)
        print(f"\nwrote {OUT} and runs/results/nn2_summary{args.tag}.json")

if __name__ == "__main__":
    main()
