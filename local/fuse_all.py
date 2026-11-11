#!/usr/bin/env python3
"""Fuse the structure stream and the new sequence stream, 8 datasets.

Input : runs/results/preds_seed/<init>/{STAG,Contacts,CNN_3D}__{DS}_fold{k}.csv
        runs/results/preds_seq/<init>/{SeqLR,SeqGBM,SeqMLP,SeqKNN}__{DS}_fold{k}.csv
        runs/results/preds_con/<init>/{ConGBM,ConLR}__{DS}_fold{k}.csv
        runs/results/preds_ext/<init>/{SeqExt,SeqExtOnly}__{DS}_fold{k}.csv
        runs/results/preds_clean/<init>/SeqClean__{DS}_fold{k}.csv
Output: stdout tables + runs/results/fusion_all.json

Two things are done more carefully here than in fuse_seeds.py:

1. `d'` fusion weights are estimated from the OTHER folds, never from the fold
   being scored. The earlier script used all five folds for the weights, which
   let a trace of the test fold into them.
2. The headline number comes from a rule CHOSEN on the other folds too
   (`selected`). Picking the best-looking rule after seeing its final AUC is the
   same selection-on-test bias this project has already been caught by once.
"""
import csv, glob, json, math, os, pathlib, sys
from collections import defaultdict
from statistics import mean, stdev

HERE = pathlib.Path(__file__).resolve().parent.parent
TREES = {"struct": HERE / "runs" / "results" / "preds_seed",
         "seq":    HERE / "runs" / "results" / "preds_seq",
         "con":    HERE / "runs" / "results" / "preds_con",
         "ext":    HERE / "runs" / "results" / "preds_ext",
         "clean":  HERE / "runs" / "results" / "preds_clean",
         "nn2":    HERE / "runs" / "results" / "preds_nn2",
         "esm":    HERE / "runs" / "results" / "preds_esm",
         "pu":     HERE / "runs" / "results" / "preds_pu",
         "dedup":  HERE / "runs" / "results" / "preds_dedup",
         # 2026-11-11: two new structure encoders trained on SLURM cluster at k=5 on
         # the same folds (verified 9,343/9,343 index+label on all 8 datasets).
         # Both LOSE to STAG alone on 8/8 datasets, and both still pay in the
         # fusion -- at k=10 they bought +0.0094, all of it on the hard sets.
         "k5new":  HERE / "runs" / "results" / "preds_k5"}
# The two k=5 encoders are OFF by default. They were added expecting the
# +0.0094 they bought in the k=10 matrix; measured at k=5 on the headline rule
# they gave -0.0013, i.e. nothing. A change that fails its own test is not
# adopted just because the members exist. Re-run the test with
#   STAG_INCLUDE_K5NEW=1 /usr/bin/python3 local/fuse_all.py
K5NEW = ["STAG_EGNN", "STAG_DUAL"] if os.environ.get("STAG_INCLUDE_K5NEW") else []
STRUCT_M = ["STAG", "Contacts", "CNN_3D"] + K5NEW
SEQ_M    = ["SeqLR", "SeqGBM", "SeqMLP", "SeqKNN"]
CON_M    = ["ConGBM", "ConLR"]
# The data stream: a second SOURCE of labels (curated published TCRs) and a
# training-set denoiser. Everything above re-reads the same 10x experiment.
# SeqDedup applies the exact-duplicate part of the filter the STAG paper says
# it applied, to the training folds only. SeqPU treats the negatives as
# unlabelled, which is what the negative-provenance analysis says they are.
# SeqNoDedup is SeqDedup's control and is deliberately NOT a member.
DATA_M   = ["SeqExt", "SeqExtOnly", "SeqClean", "SeqNN2", "SeqDedup",
            "SeqPU", "SeqAvg3"]
# Fifth stream: a protein language model. Outside knowledge about protein
# sequence in general, which none of the other four streams has.
ESM_M    = ["SeqESM", "SeqESMNN"]
ALL_M    = STRUCT_M + SEQ_M + CON_M + DATA_M + ESM_M
BAR, TCRIT = 0.75, 2.776
HLA = {"ELAGIGILTV": "A*02", "NLVPMVATV": "A*02", "GILGFVFTL": "A*02", "GLCTLVAML": "A*02",
       "RAKFKQLL": "B*08", "KLGGALQAK": "A*03", "AVFDRKSDAK": "A*11", "IVTDFSVIK": "A*11"}
SUSPECT = {d for d, h in HLA.items() if h in ("A*03", "A*11")}
RULES = ("soft", "dprime", "maxconf", "stacking")

# ---- load ------------------------------------------------------------------
P = defaultdict(dict)
for tree in TREES.values():
    for f in glob.glob(str(tree / "*" / "*_fold*.csv")):
        init = os.path.basename(os.path.dirname(f))
        tag, fold = os.path.basename(f)[:-4].rsplit("_fold", 1)
        meth, ds = tag.split("__")
        for r in csv.DictReader(open(f)):
            P[(meth, init, ds, int(fold))][str(int(float(r["index"])))] = (
                float(r["label"]), float(r["pred"]))

INITS = sorted({i for (_, i, _, _) in P})
DATASETS = sorted({d for (_, _, d, _) in P})
if not DATASETS:
    sys.exit("no predictions found")

# ---- helpers (same arithmetic as fuse_seeds.py) ----------------------------
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

def fit_logistic(X, y, iters=400, lr=1.0, l2=1e-3):
    """Same full-batch gradient descent as fuse_seeds.py, vectorised.

    21 members x 5 folds x 8 datasets x 3 pools is ~120 fits; the pure-Python
    version of this took minutes each. Mild L2 keeps it stable now that the
    member count exceeds what the earlier 9-member version had to handle.
    """
    import numpy as _np
    Xa = _np.asarray(X, dtype=_np.float64)
    ya = _np.asarray(y, dtype=_np.float64)
    Xa = _np.hstack([_np.ones((Xa.shape[0], 1)), Xa])
    w = _np.zeros(Xa.shape[1])
    n = Xa.shape[0]
    for _ in range(iters):
        p = 1.0 / (1.0 + _np.exp(-_np.clip(Xa @ w, -30, 30)))
        g = Xa.T @ (p - ya) / n
        g[1:] += l2 * w[1:]
        w -= lr * g
    return w.tolist()

# ---- per dataset -----------------------------------------------------------
report = {}
for ds in DATASETS:
    folds = sorted({f for (_, _, d, f) in P if d == ds})
    members = [(m, i) for m in ALL_M for i in INITS
               if all((m, i, ds, f) in P for f in folds)]
    if len(members) < 2:
        report[ds] = {"skipped": f"{len(members)} members"}; continue

    per_fold, folds_equal = {}, True
    for f in folds:
        sets = [set(P[ms + (ds, f)]) for ms in members]
        common = set.intersection(*sets)
        if any(len(x) != len(common) for x in sets):
            folds_equal = False
            worst = min(len(x) for x in sets)
            if len(common) < 0.9 * worst:
                report[ds] = {"skipped": "fold sets differ between members "
                                         f"(intersection {len(common)} of {worst})"}
                break
        idx = sorted(common)
        lab = [P[members[0] + (ds, f)][i][0] for i in idx]
        raw = {ms: [P[ms + (ds, f)][i][1] for i in idx] for ms in members}
        per_fold[f] = (lab, {ms: ranknorm(raw[ms]) for ms in members})
    if ds in report and "skipped" in report[ds]:
        continue

    mem_auc = {ms: [auc(list(zip(per_fold[f][0], per_fold[f][1][ms]))) for f in folds]
               for ms in members}
    bad = [ms for ms, v in mem_auc.items() if any(x != x for x in v)]
    for ms in bad: del mem_auc[ms]
    members = [ms for ms in members if ms not in bad]

    by_method = {}
    for m in ALL_M:
        v = [mean(mem_auc[(m, i)]) for i in INITS if (m, i) in mem_auc]
        if v:
            allf = [a for i in INITS if (m, i) in mem_auc for a in mem_auc[(m, i)]]
            by_method[m] = {"per_init": v, "mean": mean(v), "spread": max(v) - min(v),
                            "median": sorted(allf)[len(allf) // 2],
                            "group": ("struct" if m in STRUCT_M
                                      else "seq" if m in SEQ_M
                                      else "con" if m in CON_M
                                      else "esm" if m in ESM_M else "data")}

    POOLS = {"struct":    [ms for ms in members if ms[0] in STRUCT_M],
             "seq":       [ms for ms in members if ms[0] in SEQ_M],
             "con":       [ms for ms in members if ms[0] in CON_M],
             "data":      [ms for ms in members if ms[0] in DATA_M],
             "esm":       [ms for ms in members if ms[0] in ESM_M],
             "structseq": [ms for ms in members if ms[0] in STRUCT_M + SEQ_M],
             "all":       members}

    # (pool name, rule, fold) -> {member: weight}. Recorded so the question
    # "which models does the ensemble actually use" is answered from the run
    # rather than from the pool definition. Oat asked, 2026-11-11.
    WEIGHTS = {}

    def score(f, kind, pool, other, pname=None):
        """Scores for fold f. Every weight is estimated from `other` folds only."""
        lab, rk = per_fold[f]; n = len(lab)
        if kind == "soft":
            return [mean(rk[ms][i] for ms in pool) for i in range(n)]
        if kind == "dprime":
            w = {ms: max(dprime(mean(mem_auc[ms][g] for g in other)), 0.0) for ms in pool}
            tot = sum(w.values()) or 1.0
            if pname:
                WEIGHTS[(pname, "dprime", f)] = {f"{m}|{i}": v / tot for (m, i), v in w.items()}
            return [sum(w[ms] * rk[ms][i] for ms in pool) / tot for i in range(n)]
        if kind == "maxconf":
            return [rk[max(pool, key=lambda ms: abs(rk[ms][i] - 0.5))][i] for i in range(n)]
        if kind == "stacking":
            X, y = [], []
            for g in other:
                l2, r2 = per_fold[g]
                for i in range(len(l2)):
                    X.append([r2[ms][i] for ms in pool]); y.append(1.0 if l2[i] == 1 else 0.0)
            w = fit_logistic(X, y)
            if pname:
                WEIGHTS[(pname, "stacking", f)] = {
                    f"{pool[j][0]}|{pool[j][1]}": float(w[j + 1]) for j in range(len(pool))}
            return [w[0] + sum(w[j + 1] * rk[pool[j]][i] for j in range(len(pool)))
                    for i in range(n)]
        raise ValueError(kind)

    fused = {}
    idxf = {f: [g for g in folds if g != f] for f in folds}
    grid = {}                      # (pool, rule) -> per-fold auc
    cand_s = {}                    # (pool, rule) -> fold -> scores
    for pname, pool in POOLS.items():
        if len(pool) < 2: continue
        sc = {k: {f: score(f, k, pool, idxf[f], pname) for f in folds} for k in RULES}
        per_rule = {k: [auc(list(zip(per_fold[f][0], sc[k][f]))) for f in folds]
                    for k in RULES}
        for k, v in per_rule.items():
            fused[f"{pname}_{k}"] = v
            grid[(pname, k)] = v
            cand_s[(pname, k)] = sc[k]
        # rule chosen on the other folds, pool fixed
        sel, picks = [], []
        for f in folds:
            pos = [folds.index(x) for x in idxf[f]]
            best = max(RULES, key=lambda k: mean(per_rule[k][g] for g in pos))
            picks.append(best); sel.append(per_rule[best][folds.index(f)])
        fused[f"{pname}_selected"] = sel
        report.setdefault(ds, {}).setdefault("_picks", {})[pname] = picks

    # BOTH the pool and the rule chosen on the other folds. This is the honest
    # version of "decide per dataset": AVFDRKSDAK's sequence members sit below
    # chance, so pooling them in blind costs real AUC there, while KLGGALQAK and
    # NLVPMVATV want opposite pools. Nothing here ever reads the scored fold.
    if grid:
        sel, picks = [], []
        for f in folds:
            pos = [folds.index(x) for x in idxf[f]]
            best = max(grid, key=lambda pk: mean(grid[pk][g] for g in pos))
            picks.append(f"{best[0]}/{best[1]}")
            sel.append(grid[best][folds.index(f)])
        fused["auto"] = sel
        report.setdefault(ds, {}).setdefault("_picks", {})["auto"] = picks

        # Picking ONE winner out of ~30 candidates on four folds is itself noisy.
        # Averaging the candidates the other folds liked best spends that
        # variance instead of betting on it. Scores are rank-normalised first
        # because stacking returns logits and soft returns means.
        TOPM = 5
        comm, comm_top = [], []
        for f in folds:
            pos = [folds.index(x) for x in idxf[f]]
            top = sorted(grid, key=lambda pk: -mean(grid[pk][g] for g in pos))[:TOPM]
            # Which pools the committee actually averaged, so "how many streams
            # does the ensemble use" can be answered from the output instead of
            # guessed. Oat asked, 2026-11-11.
            comm_top.append([f"{a}/{b}" for a, b in top])
            rn = [ranknorm(cand_s[c][f]) for c in top]
            agg = [mean(v[i] for v in rn) for i in range(len(rn[0]))]
            comm.append(auc(list(zip(per_fold[f][0], agg))))
        fused["committee"] = comm
        report.setdefault(ds, {}).setdefault("_picks", {})["committee_top"] = comm_top

        # Drop members the other folds show at or below chance, then weight the
        # rest by d'. A member at 0.5 already gets weight 0 from d'; one at 0.45
        # is actively harmful and only screening removes it.
        scr = []
        for f in folds:
            pos = [folds.index(x) for x in idxf[f]]
            keep = [ms for ms in members
                    if mean(mem_auc[ms][g] for g in pos) >= 0.52] or members
            if len(keep) < 2: keep = members
            lab, rk = per_fold[f]
            w = {ms: max(dprime(mean(mem_auc[ms][g] for g in pos)), 0.0) for ms in keep}
            tot = sum(w.values()) or 1.0
            agg = [sum(w[ms] * rk[ms][i] for ms in keep) / tot for i in range(len(lab))]
            scr.append(auc(list(zip(lab, agg))))
        fused["screened"] = scr

        # The headline selector, now choosing among the pool x rule grid PLUS
        # those two. Still decided on the other folds only.
        grid2 = dict(grid)
        grid2[("*", "committee")] = comm
        grid2[("*", "screened")] = scr
        # The weights behind the combo auto2 actually chose, per fold.
        def used_weights(tag, f):
            pool, _, rule = tag.partition("/")
            return WEIGHTS.get((pool, rule, f))

        sel2, picks2 = [], []
        for f in folds:
            pos = [folds.index(x) for x in idxf[f]]
            best = max(grid2, key=lambda pk: mean(grid2[pk][g] for g in pos))
            picks2.append(f"{best[0]}/{best[1]}")
            sel2.append(grid2[best][folds.index(f)])
        fused["auto2"] = sel2
        report.setdefault(ds, {}).setdefault("_picks", {})["auto2"] = picks2
        report[ds]["_picks"]["auto2_weights"] = [used_weights(picks2[j], folds[j])
                                                 for j in range(len(folds))]

    best_mem = max(members, key=lambda ms: mean(mem_auc[ms]))
    best_struct = max((ms for ms in members if ms[0] in STRUCT_M),
                      key=lambda ms: mean(mem_auc[ms]))
    r = report.setdefault(ds, {})
    r.update({
        "hla": HLA.get(ds, "?"), "folds_equal": folds_equal, "n_members": len(members),
        "by_method": by_method,
        "best_member": f"{best_mem[0]}|{best_mem[1]}", "best_member_auc": mean(mem_auc[best_mem]),
        "best_struct": f"{best_struct[0]}|{best_struct[1]}",
        "best_struct_auc": mean(mem_auc[best_struct]),
        # per-fold values kept so the same numbers can be summarised the way the
        # original paper does (median of the cross-validation runs) as well as
        # by the mean -- comparing our mean against their median is not like for
        # like, and the low fold on a small dataset pulls a mean further than a
        # median.
        "fusion": {k: {"auc": mean(v), "median": sorted(v)[len(v) // 2], "folds": v,
                       "sd": stdev(v),
                       "vs_best_member": mean(v) - mean(mem_auc[best_mem]),
                       "vs_best_struct": mean(v) - mean(mem_auc[best_struct]),
                       "t": paired_t(v, mem_auc[best_mem])[1],
                       "t_vs_struct": paired_t(v, mem_auc[best_struct])[1],
                       "significant": abs(paired_t(v, mem_auc[best_mem])[1]) > TCRIT,
                       "sig_vs_struct": abs(paired_t(v, mem_auc[best_struct])[1]) > TCRIT}
                   for k, v in fused.items()},
    })

# ---- print -----------------------------------------------------------------
DS = sorted([d for d in report if "skipped" not in report[d]],
            key=lambda d: -report[d]["best_member_auc"])
TRUST = [d for d in DS if d not in SUSPECT]
W = 136
print(f"inits: {', '.join(INITS)}   datasets: {len(DS)}   "
      f"members per dataset: {report[DS[0]]['n_members'] if DS else 0}")
print(f"fold sets equal across all members: {all(report[d]['folds_equal'] for d in DS)}")

print("\n" + "=" * W)
print("PER-METHOD, mean over inits +- spread      [structure stream | sequence stream]")
hdr = f"{'dataset':<12}{'HLA':>6}" + "".join(f"{m[:9]:>13}" for m in ALL_M)
print(hdr); print("-" * W)
for d in DS:
    row = f"{d:<12}{report[d]['hla']:>6}"
    for m in ALL_M:
        b = report[d]["by_method"].get(m)
        row += f"{b['mean']:>7.3f}±{b['spread']:.3f}" if b else f"{'—':>13}"
    print(row)

cols = ["struct_selected", "seq_selected", "structseq_stacking",
        "all_stacking", "all_selected", "auto"]
print("\n" + "=" * W)
print("FUSION — structure-only pool, sequence-only pool, and both together")
print(f"{'dataset':<12}{'best struct':>14}{'auc':>7} |" + "".join(f"{c.replace('_',' ')[:13]:>15}" for c in cols))
print("-" * W)
for d in DS:
    r = report[d]
    line = f"{d:<12}{r['best_struct']:>14}{r['best_struct_auc']:>7.3f} |"
    for c in cols:
        v = r["fusion"].get(c)
        if not v: line += f"{'—':>15}"; continue
        mark = "*" if v["sig_vs_struct"] and v["vs_best_struct"] > 0 else (
               "x" if v["sig_vs_struct"] and v["vs_best_struct"] < 0 else " ")
        line += f"{v['auc']:>14.3f}{mark}"
    print(line)
print("-" * W)
for label, S in ((f"MEAN all {len(DS)}", DS), (f"MEAN {len(TRUST)} trusted", TRUST)):
    line = f"{label:<12}{'':>14}{mean(report[d]['best_struct_auc'] for d in S):>7.3f} |"
    for c in cols:
        vals = [report[d]["fusion"][c]["auc"] for d in S if c in report[d]["fusion"]]
        line += f"{mean(vals):>14.3f} " if vals else f"{'—':>15}"
    print(line)
print("\n* / x  = significantly better / worse than the best STRUCTURE member (paired t, |t|>2.776)")
print("auto = pool AND rule both chosen on the other folds; per-fold picks:")
for d in DS:
    pk = report[d].get("_picks", {}).get("auto", [])
    print(f"   {d:<12} {', '.join(pk)}")
print(f"Oat's bar: {BAR}   ('selected' = fusion rule chosen on the other folds, never on the scored fold)")

out = HERE / "runs" / "results" / "fusion_all.json"
out.write_text(json.dumps(report, indent=1), encoding="utf-8")
print(f"\nwrote {out}")
