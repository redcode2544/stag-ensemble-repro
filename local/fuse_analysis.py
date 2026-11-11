#!/usr/bin/env python3
"""Measure prediction-level ensemble rules instead of estimating them.

Input : runs/results/preds/{METHOD}__{DATASET}_fold{i}.csv  (index,label,pred,fold)
        produced by job 85288 with STAG_DUMP set.
Output: stdout table + runs/results/fusion.json

Answers three things the binormal estimate could only assume:
  1. do the three methods actually share the same CV folds?
  2. what is the real error correlation between them (not the assumed rho=0.5)?
  3. does any per-sample fusion rule — including Oat's max-confidence idea —
     beat the single best method, and by more than the fold-to-fold noise?

Scale trap this script exists to avoid: Contacts returns predict_proba in [0,1]
while STAG and CNN_3D return raw logits. Every rule below is therefore computed
on per-fold rank-normalised scores unless the rule is explicitly scale-free.
"""
import csv, glob, json, math, os, pathlib, sys
from collections import defaultdict
from statistics import mean, stdev

HERE = pathlib.Path(__file__).resolve().parent.parent
PRED = HERE / "runs" / "results" / "preds"
METHODS = ["STAG", "Contacts", "CNN_3D"]
TCRIT = 2.776

# ---- load -----------------------------------------------------------------
# P[(method, dataset, fold)] = {index: (label, raw_score)}
P = defaultdict(dict)
for f in glob.glob(str(PRED / "*_fold*.csv")):
    base = os.path.basename(f)[:-4]
    tag, fold = base.rsplit("_fold", 1)
    meth, ds = tag.split("__")
    for r in csv.DictReader(open(f)):
        P[(meth, ds, int(fold))][r["index"]] = (float(r["label"]), float(r["pred"]))

DATASETS = sorted({d for (_, d, _) in P})
if not DATASETS:
    sys.exit(f"no prediction files in {PRED} — copy them from SLURM cluster ~/stag/preds first")

# ---- metrics --------------------------------------------------------------
def auc(pairs):
    """ROC-AUC by rank (ties averaged). pairs = [(label, score), ...]"""
    pos = [s for l, s in pairs if l == 1]; neg = [s for l, s in pairs if l != 1]
    if not pos or not neg: return float("nan")
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][1])
    ranks = [0.0] * len(pairs); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and pairs[order[j + 1]][1] == pairs[order[i]][1]: j += 1
        r = (i + j) / 2.0 + 1
        for k in range(i, j + 1): ranks[order[k]] = r
        i = j + 1
    rsum = sum(ranks[i] for i in range(len(pairs)) if pairs[i][0] == 1)
    return (rsum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

def ranknorm(scores):
    """Per-fold rank -> (0,1). Makes logits and probabilities comparable."""
    n = len(scores)
    order = sorted(range(n), key=lambda i: scores[i])
    out = [0.0] * n
    for pos, i in enumerate(order): out[i] = (pos + 1) / (n + 1)
    return out

def pearson(a, b):
    ma, mb = mean(a), mean(b)
    va = sum((x - ma) ** 2 for x in a); vb = sum((x - mb) ** 2 for x in b)
    if va == 0 or vb == 0: return float("nan")
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)

def phi(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))
def phinv(p):
    lo, hi = -8.0, 8.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if phi(mid) < p: lo = mid
        else: hi = mid
    return (lo + hi) / 2
def dprime(a): return math.sqrt(2) * phinv(min(max(a, 1e-6), 1 - 1e-6))

def paired_t(x, y):
    d = [p - q for p, q in zip(x, y)]
    m = mean(d); s = stdev(d) if len(d) > 1 else 0.0
    return m, (float("inf") if s == 0 else m / (s / math.sqrt(len(d))))

# ---- per dataset ----------------------------------------------------------
report = {}
for ds in DATASETS:
    folds = sorted({f for (m, d, f) in P if d == ds})
    have = [m for m in METHODS if all((m, ds, f) in P for f in folds)]
    if not folds:
        report[ds] = {"skipped": "no folds"}; continue
    if len(have) < 2:
        report[ds] = {"skipped": f"only {have} present"}; continue

    # --- 1. do the methods share folds? check by the actual index sets ---
    fold_match, notes = True, []
    for f in folds:
        sets = {m: set(P[(m, ds, f)]) for m in have}
        ref = sets[have[0]]
        for m in have[1:]:
            if sets[m] != ref:
                fold_match = False
                notes.append(f"fold {f}: {m} has {len(sets[m])} ids vs {have[0]} {len(ref)}, "
                             f"overlap {len(sets[m] & ref)}")
    # fuse only on indices every method scored in that fold
    per_fold = {}
    for f in folds:
        common = set.intersection(*[set(P[(m, ds, f)]) for m in have])
        idx = sorted(common)
        lab = [P[(have[0], ds, f)][i][0] for i in idx]
        raw = {m: [P[(m, ds, f)][i][1] for i in idx] for m in have}
        per_fold[f] = (idx, lab, raw, {m: ranknorm(raw[m]) for m in have})

    # --- 2. single-method AUC, recomputed from the dumped predictions ---
    single = {m: [auc(list(zip(per_fold[f][1], per_fold[f][2][m]))) for f in folds] for m in have}
    best = max(have, key=lambda m: mean(single[m]))

    # --- 3. measured error correlation on rank scores ---
    rhos = {}
    for a in range(len(have)):
        for b in range(a + 1, len(have)):
            ma, mb = have[a], have[b]
            rr = [pearson(per_fold[f][3][ma], per_fold[f][3][mb]) for f in folds]
            rhos[f"{ma}|{mb}"] = mean(rr)

    # --- 4. the fusion rules ---
    def rule_scores(f, kind):
        idx, lab, raw, rk = per_fold[f]
        n = len(idx)
        if kind == "soft_equal":      return [mean(rk[m][i] for m in have) for i in range(n)]
        if kind == "rank_avg":        return [mean(rk[m][i] for m in have) for i in range(n)]
        if kind == "dprime_weighted":
            w = {m: max(dprime(mean(single[m])), 0.0) for m in have}
            tot = sum(w.values()) or 1.0
            return [sum(w[m] * rk[m][i] for m in have) / tot for i in range(n)]
        if kind == "max_confidence":
            # Oat's idea: trust whichever stream is most certain about THIS sample.
            # Certainty on a rank scale = distance from the middle (0.5).
            out = []
            for i in range(n):
                m_best = max(have, key=lambda m: abs(rk[m][i] - 0.5))
                out.append(rk[m_best][i])
            return out
        if kind == "max_raw":
            # the literal "highest score wins" on raw outputs, to show the scale trap
            return [max(raw[m][i] for m in have) for i in range(n)]
        raise ValueError(kind)

    RULES = ["soft_equal", "dprime_weighted", "max_confidence", "max_raw"]
    fused = {k: [auc(list(zip(per_fold[f][1], rule_scores(f, k)))) for f in folds] for k in RULES}

    # --- 5. stacking: logistic regression on the 3 rank scores, nested by fold ---
    def fit_logistic(X, y, iters=300, lr=0.5):
        w = [0.0] * (len(X[0]) + 1)
        for _ in range(iters):
            g = [0.0] * len(w)
            for xi, yi in zip(X, y):
                z = w[0] + sum(w[j + 1] * xi[j] for j in range(len(xi)))
                p = 1 / (1 + math.exp(-max(-30, min(30, z))))
                e = p - yi
                g[0] += e
                for j in range(len(xi)): g[j + 1] += e * xi[j]
            for j in range(len(w)): w[j] -= lr * g[j] / len(X)
        return w
    stack = []
    for f in folds:                              # train on the other folds only
        tr_X, tr_y = [], []
        for g in folds:
            if g == f: continue
            _, lab, _, rk = per_fold[g]
            for i in range(len(lab)):
                tr_X.append([rk[m][i] for m in have]); tr_y.append(1.0 if lab[i] == 1 else 0.0)
        w = fit_logistic(tr_X, tr_y)
        _, lab, _, rk = per_fold[f]
        sc = [w[0] + sum(w[j + 1] * rk[have[j]][i] for j in range(len(have)))
              for i in range(len(lab))]
        stack.append(auc(list(zip(lab, sc))))
    fused["stacking"] = stack

    # --- 6. is any rule better than the best single method? ---
    verdict = {}
    for k, v in fused.items():
        diff, t = paired_t(v, single[best])
        verdict[k] = {"auc": mean(v), "sd": stdev(v), "vs_best": diff, "t": t,
                      "significant": abs(t) > TCRIT}
    report[ds] = {"n_folds": len(folds), "methods": have, "folds_match": fold_match,
                  "fold_notes": notes[:4],
                  "single": {m: {"auc": mean(single[m]), "sd": stdev(single[m])} for m in have},
                  "best_single": best, "rho_measured": rhos, "fusion": verdict}

# ---- print ----------------------------------------------------------------
print("=" * 100)
print("FOLD AGREEMENT ACROSS METHODS (the paired tests depend on this)")
for ds, r in report.items():
    if "skipped" in r: print(f"  {ds:<12} SKIPPED: {r['skipped']}"); continue
    print(f"  {ds:<12} {'identical' if r['folds_match'] else 'DIFFERENT'}"
          + ("" if r["folds_match"] else "  " + "; ".join(r["fold_notes"])))

print("\n" + "=" * 100)
print("MEASURED error correlation between methods (binormal estimate assumed rho=0.5)")
for ds, r in report.items():
    if "skipped" in r: continue
    print(f"  {ds:<12} " + "  ".join(f"{k}={v:+.3f}" for k, v in r["rho_measured"].items()))

print("\n" + "=" * 100)
print("FUSION RULES vs the best single method  (paired over folds, |t|>2.776 = p<0.05)")
hdr = f"{'dataset':<12}{'best single':>13}{'auc':>8} |"
for k in ["soft_equal", "dprime_weighted", "max_confidence", "max_raw", "stacking"]:
    hdr += f"{k[:9]:>11}"
print(hdr)
print("-" * 100)
for ds, r in report.items():
    if "skipped" in r: continue
    line = f"{ds:<12}{r['best_single']:>13}{r['single'][r['best_single']]['auc']:>8.4f} |"
    for k in ["soft_equal", "dprime_weighted", "max_confidence", "max_raw", "stacking"]:
        v = r["fusion"].get(k)
        if not v: line += f"{'-':>11}"; continue
        mark = "*" if v["significant"] and v["vs_best"] > 0 else (
               "x" if v["significant"] and v["vs_best"] < 0 else " ")
        line += f"{v['auc']:>10.4f}{mark}"
    print(line)
print("-" * 100)
print("* = significantly better than the best single method   x = significantly worse")

out = HERE / "runs" / "results" / "fusion.json"
out.write_text(json.dumps(report, indent=1), encoding="utf-8")
print(f"\nwrote {out}")
