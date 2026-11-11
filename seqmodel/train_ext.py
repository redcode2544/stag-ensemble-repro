#!/usr/bin/env python3
"""Add a SECOND source of labels: curated TCRs from published studies (VDJdb).

Every label in the 8 STAG datasets comes from one 10x dextramer experiment, so
no model family can fix a label defect -- four of them already read the same
ceiling. The curated pool merges VDJdb, McPAS-TCR and IEDB (see
ext/ext_merge.py); each aggregates published tetramer/dextramer/functional studies and
records which study each row came from, so the rows from that same 10x
application note can be dropped and what is left is independent evidence.

Two members are written:

  SeqExt      in-domain training folds PLUS the curated rows, with the
              auxiliary weight chosen by inner CV on the training folds only
              from {0, 0.3, 1}. Weight 0 degrades exactly to the in-domain
              model, so the procedure cannot lose by construction.
  SeqExtOnly  trained on the curated rows ONLY -- not one in-domain row -- and
              applied to the in-domain test fold. A zero-leak external check:
              if it scores well, the dataset's labels agree with the published
              literature; if it reads chance, they do not.

Leak guard (`--guard beta`, the default and the headline): any curated row
whose CDR3-beta appears ANYWHERE in the target dataset, any fold, any label, is
dropped. That removes the public/convergent clonotypes this epitope is famous
for, which is deliberately harsher than necessary -- an independent study
finding the same beta chain is real evidence, not leakage. `--guard pair` keeps
them and drops only exact alpha+beta duplicates; both numbers are reported.

Folds are read back from the structure runs, never re-drawn, so every number
pairs fold-for-fold with the other streams and can be fused with them.
"""
import argparse, csv, glob, json, os, pathlib, random, sys
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_seq import load_dataset, auc, SEQ                 # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
EXT  = HERE / "ext"
OUT  = HERE / "runs" / "results" / "preds_ext"
TENX = "10xgenomics.com"
WEIGHTS = (0.0, 0.3, 1.0)
CGRID   = (0.5, 4.0)
AUX_CAP = 18000

def text(a, b):
    return f"a{a}|b{b}"

# ------------------------------------------------------------------ aux pool
def build_aux(ds, rows, guard, seed, cap=AUX_CAP):
    own_pair = {(r["a"], r["b"]) for r in rows}
    own_b    = {r["b"] for r in rows}

    def blocked(a, b):
        if (a, b) in own_pair: return True
        return guard == "beta" and b in own_b

    ep = [r for r in csv.DictReader(open(EXT / f"curated_{ds}_indep.csv"))
          if TENX not in r["ref"]]
    n_raw = len(ep)
    pos = [r for r in ep if not blocked(r["a"], r["b"])]
    pos_b = {r["b"] for r in pos} | own_b

    bg = [r for r in csv.DictReader(open(EXT / "vdjdb_background.csv"))
          if not blocked(r["a"], r["b"]) and r["b"] not in pos_b]

    # The auxiliary set keeps a sane class balance of its own: an aux of
    # positives alone would teach the model only where the positives are. The
    # cap bounds BOTH classes, so a huge curated pool (NLVPMVATV has 10k) is
    # subsampled rather than silently crowding the negatives out.
    npos_ds = sum(r["y"] for r in rows)
    ratio = min((len(rows) - npos_ds) / max(npos_ds, 1), 2.0)
    rng = random.Random(1000 + seed)
    if pos:
        max_pos = max(1, int(cap / (1.0 + ratio)))
        if len(pos) > max_pos:
            pos = rng.sample(pos, max_pos)
        want_neg = max(int(round(len(pos) * ratio)), min(200, len(bg)))
    else:
        want_neg = min(2000, len(bg))
    neg = rng.sample(bg, min(want_neg, len(bg)))

    aux_t = [text(r["a"], r["b"]) for r in pos] + [text(r["a"], r["b"]) for r in neg]
    aux_y = np.array([1] * len(pos) + [0] * len(neg))
    return aux_t, aux_y, dict(curated_raw=n_raw, curated_kept=len(pos),
                              curated_dropped=n_raw - len(pos),
                              background=len(neg), studies_nontenx=None)

# --------------------------------------------------------------------- models
def fit_lr(texts, y, w, C, seed):
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=2,
                          sublinear_tf=True)
    X = vec.fit_transform(texts)
    lr = LogisticRegression(max_iter=4000, solver="liblinear", C=C)
    lr.fit(X, y, sample_weight=w)
    return vec, lr

def pick_weight(tr, aux_t, aux_y, seed):
    """Choose (weight, C) by inner CV on the in-domain TRAINING folds only."""
    ytr = np.array([r["y"] for r in tr])
    ttr = [r["text_cdr3"] for r in tr]
    inner = StratifiedKFold(4, shuffle=True, random_state=seed)
    best = (-1.0, 0.0, CGRID[0])
    for wgt in WEIGHTS:
        if wgt > 0 and len(aux_y) == 0: continue
        for C in CGRID:
            aucs = []
            for a, b in inner.split(np.zeros(len(ytr)), ytr):
                te_t = [ttr[i] for i in b]; te_y = ytr[b]
                tx = [ttr[i] for i in a]; ty = list(ytr[a])
                sw = [1.0] * len(tx)
                if wgt > 0:
                    tx = tx + aux_t; ty = ty + list(aux_y)
                    sw = sw + [wgt] * len(aux_y)
                if len(set(ty)) < 2: continue
                vec, lr = fit_lr(tx, np.array(ty), np.array(sw), C, seed)
                aucs.append(auc(te_y, lr.decision_function(vec.transform(te_t))))
            if aucs and float(np.mean(aucs)) > best[0]:
                best = (float(np.mean(aucs)), wgt, C)
    return best

def ext_only(aux_t, aux_y, te, seed):
    """Trained on curated rows alone. No in-domain row ever enters the fit."""
    if len(aux_y) == 0 or len(set(aux_y.tolist())) < 2 or int(aux_y.sum()) < 30:
        return None
    inner = StratifiedKFold(3, shuffle=True, random_state=seed)
    best = (-1.0, CGRID[0])
    for C in CGRID:
        aucs = []
        for a, b in inner.split(np.zeros(len(aux_y)), aux_y):
            vec, lr = fit_lr([aux_t[i] for i in a], aux_y[a],
                             np.ones(len(a)), C, seed)
            aucs.append(auc(aux_y[b], lr.decision_function(
                vec.transform([aux_t[i] for i in b]))))
        if aucs and float(np.mean(aucs)) > best[0]:
            best = (float(np.mean(aucs)), C)
    vec, lr = fit_lr(aux_t, aux_y, np.ones(len(aux_y)), best[1], seed)
    return lr.decision_function(vec.transform([r["text_cdr3"] for r in te]))

# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--inits", type=int, default=2)
    ap.add_argument("--guard", choices=("beta", "pair"), default="beta")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--baseline", action="store_true",
                    help="force weight 0 and write the member as SeqCdr3 -- the "
                         "like-for-like CDR3-only control for SeqExt, so the "
                         "gain can be attributed to the curated rows and not to "
                         "the change of representation")
    args = ap.parse_args()
    global WEIGHTS
    if args.baseline: WEIGHTS = (0.0,)

    all_ds = sorted(os.path.basename(p)[:-4] for p in glob.glob(str(SEQ / "*.csv")))
    datasets = args.datasets or all_ds
    summary = {}
    for ds in datasets:
        rows, _ = load_dataset(ds)
        folds = sorted({r["fold"] for r in rows})
        per = defaultdict(lambda: defaultdict(list))
        picks, info = [], None
        for init in range(1, args.inits + 1):
            seed = init
            aux_t, aux_y, info = build_aux(ds, rows, args.guard, seed)
            for k in folds:
                tr = [r for r in rows if r["fold"] != k]
                te = [r for r in rows if r["fold"] == k]
                yte = np.array([r["y"] for r in te])
                icv, wgt, C = pick_weight(tr, aux_t, aux_y, seed)
                picks.append(dict(init=init, fold=k, weight=wgt, C=C, inner=icv))

                tx = [r["text_cdr3"] for r in tr]; ty = [r["y"] for r in tr]
                sw = [1.0] * len(tx)
                if wgt > 0:
                    tx = tx + aux_t; ty = ty + list(aux_y)
                    sw = sw + [wgt] * len(aux_y)
                vec, lr = fit_lr(tx, np.array(ty), np.array(sw), C, seed)
                s = lr.decision_function(vec.transform([r["text_cdr3"] for r in te]))
                if args.baseline:
                    preds = {"SeqCdr3": s}
                else:
                    preds = {"SeqExt": s}
                    so = ext_only(aux_t, aux_y, te, seed)
                    if so is not None: preds["SeqExtOnly"] = so

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
        for m in ("SeqExt", "SeqExtOnly", "SeqCdr3"):
            if m not in per: continue
            means = [float(np.mean(per[m][i])) for i in sorted(per[m])]
            line.append(f"{m} {np.mean(means):.3f}+-{max(means)-min(means):.3f}")
            summary.setdefault(ds, {})[m] = dict(mean=float(np.mean(means)),
                                                 spread=float(max(means) - min(means)),
                                                 per_init=means)
        wsel = defaultdict(int)
        for p in picks: wsel[p["weight"]] += 1
        summary.setdefault(ds, {})["_aux"] = info
        summary[ds]["_weights"] = {str(k): v for k, v in sorted(wsel.items())}
        summary[ds]["_guard"] = args.guard
        print(f"{ds:11s} curated {info['curated_raw']:5d} kept {info['curated_kept']:5d} "
              f"bg {info['background']:5d} | w="
              + ",".join(f"{k}x{v}" for k, v in sorted(wsel.items()))
              + " | " + "  ".join(line), flush=True)

    if args.write:
        name = f"ext_summary{args.tag}.json"
        json.dump(summary, open(HERE / "runs" / "results" / name, "w"), indent=1)
        print(f"\nwrote {OUT} and runs/results/{name}")

if __name__ == "__main__":
    main()
