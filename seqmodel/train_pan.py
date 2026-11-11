#!/usr/bin/env python3
"""Pan-peptide training, done so that it cannot leak.

The negatives in this benchmark are other peptides' POSITIVES: 52.8% of rows
hold a TCR that also appears in another dataset, and of the TCRs seen in three
datasets, 386 are positive in exactly one and only ONE is positive in two. So a
model trained on the pooled data can answer "is this TCR a negative for peptide
d?" with "yes, because I saw it binding peptide e" -- memorisation dressed up as
transfer. Train naively and the score is fake.

The guard here is absolute: for target dataset d, the auxiliary pool excludes
every TCR alpha-beta pair that appears ANYWHERE in d, in any fold, with any
label. Nothing in d's test fold can be recognised from the auxiliary data.

Whether the auxiliary data helps at all is decided by inner CV on the training
folds, as a sample weight in {0, 0.25, 0.5, 1.0}. Weight 0 degrades to the
in-domain model, so this can only help if the transfer is real.
"""
import csv, glob, json, os, pathlib, sys, argparse
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdr3 import cdr3                                       # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
SEQ  = HERE / "seq"
REF  = HERE / "runs" / "results" / "preds_seed" / "init1"
OUT  = HERE / "runs" / "results" / "preds_pan"
TAIL = 20
WEIGHTS = (0.0, 0.3, 1.0)

def load_folds(ds):
    fold = {}
    for f in sorted(glob.glob(str(REF / f"STAG__{ds}_fold*.csv"))):
        k = int(os.path.basename(f)[:-4].rsplit("_fold", 1)[1])
        for r in csv.DictReader(open(f)):
            fold[str(int(float(r["index"])))] = k
    return fold

def load_all():
    data = {}
    for f in sorted(glob.glob(str(SEQ / "*.csv"))):
        ds = os.path.basename(f)[:-4]
        fold = load_folds(ds)
        rows = []
        for r in csv.DictReader(open(f)):
            i = str(int(float(r["index"])))
            if i not in fold: continue
            a = cdr3(r["tcra"])[0] or r["tcra"][-TAIL:]
            b = cdr3(r["tcrb"])[0] or r["tcrb"][-TAIL:]
            rows.append(dict(index=i, y=int(float(r["label"])), fold=fold[i],
                             pep=r["pep"], a=a, b=b, ab=(r["tcra"], r["tcrb"]),
                             text=f'p{r["pep"]}|a{a}|b{b}|A{r["tcra"]}|B{r["tcrb"]}'))
        data[ds] = rows
    return data

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

def vectorise(fit_rows, apply_rows):
    """Vocabulary from `fit_rows` only -- the scored fold never enters it."""
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=2, sublinear_tf=True)
    vec.fit([r["text"] for r in fit_rows])
    return vec.transform([r["text"] for r in apply_rows])

def fit_score_X(Xtr, ytr, wtr, Xte, seed, C=1.0):
    m = LogisticRegression(max_iter=4000, solver="liblinear", C=C, random_state=seed)
    m.fit(Xtr, ytr, sample_weight=np.asarray(wtr))
    return m.decision_function(Xte)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inits", type=int, default=3)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    data = load_all()
    summary = {}
    for ds in (args.datasets or sorted(data)):
        rows = data[ds]
        own_tcrs = {r["ab"] for r in rows}
        aux = [r for e, v in data.items() if e != ds for r in v if r["ab"] not in own_tcrs]
        folds = sorted({r["fold"] for r in rows})
        per = defaultdict(list); chose = []
        for init in range(1, args.inits + 1):
            fold_auc = []
            for k in folds:
                tr = [r for r in rows if r["fold"] != k]
                te = [r for r in rows if r["fold"] == k]
                # One vectoriser per outer fold, fitted on in-domain train + aux.
                # Every inner fit then reuses slices of it, which is what makes
                # this affordable; the test fold is still never in the vocabulary.
                pool = tr + aux
                Xpool = vectorise(pool, pool)
                Xte_m = vectorise(pool, te)
                ntr = len(tr)
                Xin, Xaux = Xpool[:ntr], Xpool[ntr:]
                yin = np.array([r["y"] for r in tr])
                yaux = np.array([r["y"] for r in aux])
                inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=init)
                best = (-1.0, 0.0)
                for w in WEIGHTS:
                    sc = []
                    for ia, ib in inner.split(np.zeros(ntr), yin):
                        if w > 0:
                            from scipy.sparse import vstack
                            Xa = vstack([Xin[ia], Xaux]); ya = np.concatenate([yin[ia], yaux])
                            wa = np.concatenate([np.ones(len(ia)), np.full(len(yaux), w)])
                        else:
                            Xa, ya, wa = Xin[ia], yin[ia], np.ones(len(ia))
                        sc.append(auc(yin[ib], fit_score_X(Xa, ya, wa, Xin[ib], init)))
                    m = float(np.mean(sc))
                    if m > best[0]: best = (m, w)
                w = best[1]; chose.append(w)
                if w > 0:
                    from scipy.sparse import vstack
                    Xa = vstack([Xin, Xaux]); ya = np.concatenate([yin, yaux])
                    wa = np.concatenate([np.ones(ntr), np.full(len(yaux), w)])
                else:
                    Xa, ya, wa = Xin, yin, np.ones(ntr)
                s = fit_score_X(Xa, ya, wa, Xte_m, init)
                fold_auc.append(auc([r["y"] for r in te], s))
                if args.write:
                    d = OUT / f"init{init}"; d.mkdir(parents=True, exist_ok=True)
                    with open(d / f"PanLR__{ds}_fold{k}.csv", "w", newline="") as fh:
                        wr = csv.writer(fh); wr.writerow(["index", "label", "pred", "fold"])
                        for r, v in zip(te, s): wr.writerow([r["index"], r["y"], float(v), k])
            per["PanLR"].append(float(np.mean(fold_auc)))
        means = per["PanLR"]
        summary[ds] = dict(mean=float(np.mean(means)), spread=max(means) - min(means),
                           per_init=means, aux_rows=len(aux),
                           aux_weight_picks={str(w): chose.count(w) for w in WEIGHTS})
        print(f"{ds:12s} aux={len(aux):5d} rows | PanLR {np.mean(means):.3f}±{max(means)-min(means):.3f}"
              f" | aux weight picked " +
              ", ".join(f"{w}x{chose.count(w)}" for w in WEIGHTS if chose.count(w)), flush=True)
    if args.write:
        json.dump(summary, open(HERE / "runs" / "results" / "pan_summary.json", "w"), indent=1)
        print(f"\nwrote {OUT} and runs/results/pan_summary.json")

if __name__ == "__main__":
    main()
