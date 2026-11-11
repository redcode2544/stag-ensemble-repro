#!/usr/bin/env python3
"""Multi-seed analysis: error bars per dataset, and a 9-member ensemble.

Input : runs/results/preds_seed/<seed>/{METHOD}__{DATASET}_fold{i}.csv
        (job 85337: 8 datasets x 3 methods x 3 seeds, torch seeded)
Output: stdout tables + runs/results/fusion_seeds.json

Members are (method, seed) pairs, so fusion runs across BOTH axes. Averaging
across seeds attacks initialisation variance, which measurement showed is the
dominant error on the small datasets (STAG moved up to 0.057 AUC between
identical runs before torch was seeded).

Every rule works on per-fold rank-normalised scores: Contacts returns
predict_proba in [0,1] while STAG and CNN_3D return raw logits, so raw scores
are not comparable across methods.
"""
import csv, glob, json, math, os, pathlib, sys, itertools
from collections import defaultdict
from statistics import mean, stdev

HERE = pathlib.Path(__file__).resolve().parent.parent
PRED = HERE / "runs" / "results" / "preds_seed"
METHODS = ["STAG", "Contacts", "CNN_3D"]
BAR, TCRIT = 0.75, 2.776
HLA = {"ELAGIGILTV": "A*02", "NLVPMVATV": "A*02", "GILGFVFTL": "A*02", "GLCTLVAML": "A*02",
       "RAKFKQLL": "B*08", "KLGGALQAK": "A*03", "AVFDRKSDAK": "A*11", "IVTDFSVIK": "A*11"}
SUSPECT = {d for d, h in HLA.items() if h in ("A*03", "A*11")}

# ---- load: P[(method, seed, dataset, fold)] = {index: (label, score)} -------
P = defaultdict(dict)
for f in glob.glob(str(PRED / "*" / "*_fold*.csv")):
    seed = os.path.basename(os.path.dirname(f))
    tag, fold = os.path.basename(f)[:-4].rsplit("_fold", 1)
    meth, ds = tag.split("__")
    for r in csv.DictReader(open(f)):
        P[(meth, seed, ds, int(fold))][r["index"]] = (float(r["label"]), float(r["pred"]))

SEEDS = sorted({s for (_, s, _, _) in P})
DATASETS = sorted({d for (_, _, d, _) in P})
if not DATASETS:
    sys.exit(f"no predictions under {PRED} — copy them from SLURM cluster ~/stag/preds_seed/ first")

# ---- helpers ---------------------------------------------------------------
def auc(pairs):
    pos = sum(1 for l, _ in pairs if l == 1); neg = len(pairs) - pos
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
    return (rsum - pos * (pos + 1) / 2) / (pos * neg)

def ranknorm(v):
    n = len(v); out = [0.0] * n
    for pos, i in enumerate(sorted(range(n), key=lambda i: v[i])): out[i] = (pos + 1) / (n + 1)
    return out

def phi(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))
def phinv(p):
    lo, hi = -8.0, 8.0
    for _ in range(200):
        m = (lo + hi) / 2
        if phi(m) < p: lo = m
        else: hi = m
    return (lo + hi) / 2
def dprime(a): return math.sqrt(2) * phinv(min(max(a, 1e-6), 1 - 1e-6))

def paired_t(x, y):
    d = [a - b for a, b in zip(x, y)]
    m = mean(d); s = stdev(d) if len(d) > 1 else 0.0
    return m, (float("inf") if s == 0 else m / (s / math.sqrt(len(d))))

def fit_logistic(X, y, iters=300, lr=0.5):
    w = [0.0] * (len(X[0]) + 1)
    for _ in range(iters):
        g = [0.0] * len(w)
        for xi, yi in zip(X, y):
            z = w[0] + sum(w[j + 1] * xi[j] for j in range(len(xi)))
            e = 1 / (1 + math.exp(-max(-30, min(30, z)))) - yi
            g[0] += e
            for j in range(len(xi)): g[j + 1] += e * xi[j]
        for j in range(len(w)): w[j] -= lr * g[j] / len(X)
    return w

# ---- per dataset -----------------------------------------------------------
report = {}
for ds in DATASETS:
    folds = sorted({f for (_, _, d, f) in P if d == ds})
    members = [(m, s) for m in METHODS for s in SEEDS
               if all((m, s, ds, f) in P for f in folds)]
    if len(members) < 2:
        report[ds] = {"skipped": f"{len(members)} members"}; continue

    per_fold = {}
    ok = True
    aborted = False
    for f in folds:
        sets = [set(P[(m, s, ds, f)]) for (m, s) in members]
        common = set.intersection(*sets)
        if any(len(x) != len(common) for x in sets):
            ok = False
            # Guard against the job-85337 mistake: if the fold draw moved between
            # runs, members do not share out-of-fold samples and averaging their
            # predictions mixes training data into the test set.
            worst = min(len(x) for x in sets)
            if len(common) < 0.9 * worst:
                report[ds] = {"skipped": "fold sets differ between members "
                              f"(intersection {len(common)} of {worst}) — the split "
                              "seed must be held constant across runs"}
                break
        idx = sorted(common)
        lab = [P[members[0]+(ds, f)][i][0] for i in idx]
        raw = {ms: [P[ms + (ds, f)][i][1] for i in idx] for ms in members}
        per_fold[f] = (lab, raw, {ms: ranknorm(raw[ms]) for ms in members})
    if ds in report and "skipped" in report[ds]:
        continue

    # per-member AUC, and per-method spread across seeds (the error bar)
    mem_auc = {ms: [auc(list(zip(per_fold[f][0], per_fold[f][1][ms]))) for f in folds]
               for ms in members}
    # a fold still being written, or one holding a single class, yields nan.
    # Drop such members rather than letting the nan poison every later statistic.
    bad = [ms for ms, v in mem_auc.items() if any(x != x for x in v)]
    for ms in bad:
        del mem_auc[ms]
    members = [ms for ms in members if ms not in bad]
    if len(members) < 2:
        report[ds] = {"skipped": f"{len(members)} usable members ({len(bad)} had nan folds)"}
        continue
    by_method = {}
    for m in METHODS:
        v = [mean(mem_auc[(m, s)]) for s in SEEDS if (m, s) in mem_auc]
        if v:
            by_method[m] = {"per_seed": v, "mean": mean(v),
                            "spread": max(v) - min(v),
                            "sd": stdev(v) if len(v) > 1 else 0.0}

    def rule(f, kind, pool):
        lab, raw, rk = per_fold[f]; n = len(lab)
        if kind == "soft":
            return [mean(rk[ms][i] for ms in pool) for i in range(n)]
        if kind == "dprime":
            w = {ms: max(dprime(mean(mem_auc[ms])), 0.0) for ms in pool}
            tot = sum(w.values()) or 1.0
            return [sum(w[ms] * rk[ms][i] for ms in pool) / tot for i in range(n)]
        if kind == "maxconf":
            out = []
            for i in range(n):
                b = max(pool, key=lambda ms: abs(rk[ms][i] - 0.5))
                out.append(rk[b][i])
            return out
        raise ValueError(kind)

    results = {}
    # seed-averaging within each method (isolates the gain from seeds alone)
    for m in METHODS:
        pool = [(m, s) for s in SEEDS if (m, s) in mem_auc]
        if len(pool) > 1:
            results[f"seedavg_{m}"] = [auc(list(zip(per_fold[f][0], rule(f, "soft", pool))))
                                       for f in folds]
    # full pool, every method x every seed
    for kind, name in (("soft", "all_soft"), ("dprime", "all_dprime"), ("maxconf", "all_maxconf")):
        results[name] = [auc(list(zip(per_fold[f][0], rule(f, kind, members)))) for f in folds]
    # stacking on the full pool, trained on the other folds only
    st = []
    for f in folds:
        X, y = [], []
        for g in folds:
            if g == f: continue
            lab, _, rk = per_fold[g]
            for i in range(len(lab)):
                X.append([rk[ms][i] for ms in members]); y.append(1.0 if lab[i] == 1 else 0.0)
        w = fit_logistic(X, y)
        lab, _, rk = per_fold[f]
        st.append(auc(list(zip(lab, [w[0] + sum(w[j + 1] * rk[members[j]][i]
                                                for j in range(len(members)))
                                     for i in range(len(lab))]))))
    results["all_stacking"] = st

    best_mem = max(members, key=lambda ms: mean(mem_auc[ms]))
    report[ds] = {
        "hla": HLA.get(ds, "?"), "folds_equal": ok, "n_members": len(members),
        "members": {f"{m}|{s}": mean(v) for (m, s), v in mem_auc.items()},
        "by_method": by_method,
        "best_member": f"{best_mem[0]}|{best_mem[1]}", "best_member_auc": mean(mem_auc[best_mem]),
        "best_method_mean": max(by_method.values(), key=lambda x: x["mean"])["mean"],
        "fusion": {k: {"auc": mean(v), "sd": stdev(v),
                       "vs_best_member": mean(v) - mean(mem_auc[best_mem]),
                       "t": paired_t(v, mem_auc[best_mem])[1],
                       "significant": abs(paired_t(v, mem_auc[best_mem])[1]) > TCRIT}
                   for k, v in results.items()},
    }

# ---- print -----------------------------------------------------------------
DS = sorted([d for d in report if "skipped" not in report[d]],
            key=lambda d: -report[d]["best_member_auc"])
TRUST = [d for d in DS if d not in SUSPECT]
print(f"seeds: {', '.join(SEEDS)}   datasets: {len(DS)}   "
      f"members per dataset: {report[DS[0]]['n_members'] if DS else 0}")
print("fold sets equal across all members: "
      f"{all(report[d]['folds_equal'] for d in DS)}")

print("\n" + "=" * 104)
print("PER-METHOD, mean over seeds +- spread  (the spread IS the error bar)")
print(f"{'dataset':<12}{'HLA':>6}" + "".join(f"{m:>22}" for m in METHODS))
print("-" * 104)
for d in DS:
    row = f"{d:<12}{report[d]['hla']:>6}"
    for m in METHODS:
        b = report[d]["by_method"].get(m)
        row += f"{b['mean']:>15.3f} ±{b['spread']:>5.3f}" if b else f"{'—':>22}"
    print(row)

print("\n" + "=" * 104)
print(f"FUSION — {report[DS[0]][chr(39)+chr(39)] if False else report[DS[0]]['n_members']} members "
      f"({len(METHODS)} methods x {len(SEEDS)} inits) vs the single best member")
cols = ["seedavg_STAG", "all_soft", "all_dprime", "all_stacking", "all_maxconf"]
print(f"{'dataset':<12}{'best member':>16}{'auc':>8} |" + "".join(f"{c[:11]:>13}" for c in cols))
print("-" * 104)
for d in DS:
    r = report[d]
    line = f"{d:<12}{r['best_member']:>16}{r['best_member_auc']:>8.3f} |"
    for c in cols:
        v = r["fusion"].get(c)
        if not v: line += f"{'—':>13}"; continue
        mark = "*" if v["significant"] and v["vs_best_member"] > 0 else (
               "x" if v["significant"] and v["vs_best_member"] < 0 else " ")
        line += f"{v['auc']:>12.3f}{mark}"
    print(line)
print("-" * 104)
for label, S in (("MEAN over all 8", DS), ("MEAN over 5 trustworthy", TRUST)):
    line = f"{label:<12}{'':>16}{mean(report[d]['best_member_auc'] for d in S):>8.3f} |"
    for c in cols:
        vals = [report[d]["fusion"][c]["auc"] for d in S if c in report[d]["fusion"]]
        line += f"{mean(vals):>12.3f} " if vals else f"{'—':>13}"
    print(line)
print("\n* significantly better than the best single member   x significantly worse")
print(f"Oat's bar: {BAR}")

out = HERE / "runs" / "results" / "fusion_seeds.json"
out.write_text(json.dumps(report, indent=1), encoding="utf-8")
print(f"\nwrote {out}")
