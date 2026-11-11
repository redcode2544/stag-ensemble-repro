#!/usr/bin/env python3
"""Sequence stream for STAG's 8 peptide-specific datasets.

The three shipped methods all read the MODELLED STRUCTURE. This reads the raw
TCR sequence instead -- the same information before any modelling error is
added. Within one dataset the peptide and MHC are constant, so the task is
"which TCRs recognise this one epitope", which is what the sequence literature
(NetTCR, pMTnet, ERGO) actually solves.

Folds are not re-drawn: they are read back from the existing prediction dumps,
so every number here is paired fold-for-fold with the structure methods and can
be fused with them honestly.

No remembered numeric constants (no BLOSUM, no Atchley) -- every feature is
derived from the sequences themselves, so the whole thing is reproducible from
this file alone.

Members written:
  SeqLR   logistic regression on char n-grams of CDR3a|CDR3b   (motif detector)
  SeqGBM  gradient-boosted trees on composition + n-gram SVD   (non-linear)
  SeqMLP  small neural net on n-gram SVD                       (different bias)
  SeqKNN  cosine k-NN in n-gram space                          (instance-based)
"""
import csv, glob, json, os, pathlib, sys, argparse
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdr3 import cdr3                                    # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
SEQ  = HERE / "seq"
REF  = HERE / "runs" / "results" / "preds_seed" / "init1"   # folds live here
OUT  = HERE / "runs" / "results" / "preds_seq"
AAS  = "ACDEFGHIKLMNPQRSTVWY"
TAIL = 20          # fallback window when the CDR3 motif is not found

# ---------------------------------------------------------------- data loading
def load_folds(ds):
    """index -> fold, read back from the structure runs. Never re-drawn here."""
    fold = {}
    for f in sorted(glob.glob(str(REF / f"STAG__{ds}_fold*.csv"))):
        k = int(os.path.basename(f)[:-4].rsplit("_fold", 1)[1])
        for r in csv.DictReader(open(f)):
            fold[str(int(float(r["index"])))] = k
    return fold

def load_dataset(ds):
    rows = list(csv.DictReader(open(SEQ / f"{ds}.csv")))
    fold = load_folds(ds)
    keep, dropped = [], 0
    for r in rows:
        i = str(int(float(r["index"])))
        if i not in fold:
            dropped += 1
            continue
        a = cdr3(r["tcra"])[0] or r["tcra"][-TAIL:]
        b = cdr3(r["tcrb"])[0] or r["tcrb"][-TAIL:]
        keep.append(dict(index=i, y=int(float(r["label"])), fold=fold[i],
                         a=a, b=b, fa=r["tcra"], fb=r["tcrb"]))
    for r in keep:
        r["text_cdr3"] = f'a{r["a"]}|b{r["b"]}'
        r["text_full"] = f'A{r["fa"]}|B{r["fb"]}'
        r["text_both"] = r["text_cdr3"] + "|" + r["text_full"]
    return keep, dropped

# ------------------------------------------------------------------- features
def composition(rows):
    """Exact, derived only from the sequences: length + aa frequency per chain."""
    X = np.zeros((len(rows), 2 + 2 * len(AAS)), dtype=np.float64)
    for n, r in enumerate(rows):
        X[n, 0], X[n, 1] = len(r["a"]), len(r["b"])
        for c, (ch, off) in enumerate(((r["a"], 2), (r["b"], 2 + len(AAS)))):
            if ch:
                for j, aa in enumerate(AAS):
                    X[n, off + j] = ch.count(aa) / len(ch)
    return X

def build(train_rows, test_rows, n_svd, seed, field="text_both"):
    """Fit every transform on TRAIN ONLY, then apply to both."""
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=2,
                          sublinear_tf=True)
    Ttr = vec.fit_transform([r[field] for r in train_rows])
    Tte = vec.transform([r[field] for r in test_rows])
    k = int(min(n_svd, Ttr.shape[1] - 1, len(train_rows) - 1))
    svd = TruncatedSVD(n_components=max(k, 2), random_state=seed)
    Str = svd.fit_transform(Ttr); Ste = svd.transform(Tte)
    Ctr, Cte = composition(train_rows), composition(test_rows)
    return Ttr, Tte, np.hstack([Str, Ctr]), np.hstack([Ste, Cte]), Str, Ste

# --------------------------------------------------------------------- models
FIELDS = ("text_cdr3", "text_full", "text_both")

def pick_field(train_rows, ytr, n_svd, seed):
    """Choose the sequence representation by inner CV on the TRAINING folds only.

    Small datasets prefer CDR3 alone; large ones gain from the framework region
    too (it carries V/J gene identity). Letting a single probe decide per fold
    keeps that choice away from the test fold entirely.
    """
    inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    best = (-1.0, FIELDS[0])
    for field in FIELDS:
        vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=2,
                              sublinear_tf=True)
        T = vec.fit_transform([r[field] for r in train_rows])
        gs = GridSearchCV(LogisticRegression(max_iter=4000, solver="liblinear"),
                          {"C": [0.1, 0.5, 1.0, 4.0, 16.0]},
                          scoring="roc_auc", cv=inner, n_jobs=1).fit(T, ytr)
        if gs.best_score_ > best[0]:
            best = (float(gs.best_score_), field)
    return best

def fit_predict(Ttr, Tte, Dtr, Dte, Str, Ste, ytr, seed):
    """-> {member: test scores}. Inner CV only ever sees the training folds."""
    out = {}
    inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)

    gs = GridSearchCV(LogisticRegression(max_iter=4000, solver="liblinear"),
                      {"C": [0.1, 0.5, 1.0, 4.0, 16.0]},
                      scoring="roc_auc", cv=inner, n_jobs=1)
    gs.fit(Ttr, ytr)
    out["SeqLR"] = gs.best_estimator_.decision_function(Tte)

    gbm = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                         max_leaf_nodes=15, l2_regularization=1.0,
                                         early_stopping=True, validation_fraction=0.2,
                                         random_state=seed)
    gbm.fit(Dtr, ytr)
    out["SeqGBM"] = gbm.predict_proba(Dte)[:, 1]

    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(128, 32), alpha=1e-3, max_iter=800,
                      early_stopping=True, n_iter_no_change=25,
                      validation_fraction=0.15, random_state=seed))
    mlp.fit(Str, ytr)
    out["SeqMLP"] = mlp.predict_proba(Ste)[:, 1]

    gs = GridSearchCV(KNeighborsClassifier(metric="cosine", weights="distance"),
                      {"n_neighbors": [5, 11, 21, 41]},
                      scoring="roc_auc", cv=inner, n_jobs=1)
    gs.fit(Ttr, ytr)
    out["SeqKNN"] = gs.best_estimator_.predict_proba(Tte)[:, 1]
    return out

# ----------------------------------------------------------------------- main
def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if not pos or not neg: return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); i = 0
    ss = s[order]
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]: j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return (ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--inits", type=int, default=3)
    ap.add_argument("--svd", type=int, default=96)
    ap.add_argument("--write", action="store_true", help="dump prediction CSVs")
    args = ap.parse_args()

    all_ds = sorted(os.path.basename(p)[:-4] for p in glob.glob(str(SEQ / "*.csv")))
    datasets = args.datasets or all_ds
    summary = {}
    for ds in datasets:
        rows, dropped = load_dataset(ds)
        folds = sorted({r["fold"] for r in rows})
        per = defaultdict(lambda: defaultdict(list))        # member -> init -> fold aucs
        inner_cv, chosen = [], []
        for init in range(1, args.inits + 1):
            seed = init
            for k in folds:
                tr = [r for r in rows if r["fold"] != k]
                te = [r for r in rows if r["fold"] == k]
                ytr = np.array([r["y"] for r in tr]); yte = np.array([r["y"] for r in te])
                icv, field = pick_field(tr, ytr, args.svd, seed)
                inner_cv.append(icv); chosen.append(field)
                Ttr, Tte, Dtr, Dte, Str, Ste = build(tr, te, args.svd, seed, field)
                preds = fit_predict(Ttr, Tte, Dtr, Dte, Str, Ste, ytr, seed)
                for m, s in preds.items():
                    per[m][init].append(auc(yte, s))
                    if args.write:
                        d = OUT / f"init{init}"; d.mkdir(parents=True, exist_ok=True)
                        with open(d / f"{m}__{ds}_fold{k}.csv", "w", newline="") as fh:
                            w = csv.writer(fh); w.writerow(["index", "label", "pred", "fold"])
                            for r, v in zip(te, s):
                                w.writerow([r["index"], r["y"], float(v), k])
        line = []
        for m in ("SeqLR", "SeqGBM", "SeqMLP", "SeqKNN"):
            means = [float(np.mean(per[m][i])) for i in sorted(per[m])]
            line.append(f"{m} {np.mean(means):.3f}±{max(means)-min(means):.3f}")
            summary.setdefault(ds, {})[m] = dict(mean=float(np.mean(means)),
                                                 spread=float(max(means) - min(means)),
                                                 per_init=means)
        from collections import Counter
        pick = ",".join(f"{k.replace('text_','')}x{v}" for k, v in Counter(chosen).most_common())
        summary.setdefault(ds, {})["_repr"] = pick
        summary[ds]["_innercv"] = float(np.mean(inner_cv))
        print(f"{ds:12s} n={len(rows):5d} | innerCV {np.mean(inner_cv):.3f} | repr {pick:22s} | "
              + "  ".join(line), flush=True)
    if args.write:
        (HERE / "runs" / "results").mkdir(parents=True, exist_ok=True)
        json.dump(summary, open(HERE / "runs" / "results" / "seq_summary.json", "w"), indent=1)
        print(f"\nwrote {OUT} and runs/results/seq_summary.json")

if __name__ == "__main__":
    main()
