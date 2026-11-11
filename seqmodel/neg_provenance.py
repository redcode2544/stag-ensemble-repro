#!/usr/bin/env python3
"""What ARE the negatives, and does the measured AUC come from specificity?

Every negative in these datasets is a real TCR taken from somewhere. Where
from decides what a high AUC means:

  * a negative that is a published/observed positive of a peptide presented by
    a DIFFERENT HLA allele can be separated on HLA restriction alone -- V-gene
    usage differs -- without the model knowing anything about the peptide;
  * a negative that binds a different peptide on the SAME allele can only be
    separated by peptide specificity, which is the thing being claimed.

So: label each negative by provenance, then score the same fused predictions
twice, once against each kind of negative. The gap between the two numbers is
how much of the benchmark is HLA reading.

Fused score = mean over members and inits of the per-fold rank of the member's
score (the `soft` rule of fuse_all.py), computed here from the prediction dumps
so no retraining is involved.
"""
import csv, glob, json, os, pathlib, sys
from collections import defaultdict

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_seq import load_dataset, auc, SEQ                 # noqa: E402

HERE  = pathlib.Path(__file__).resolve().parent.parent
RES   = HERE / "runs" / "results"
TREES = {"struct": RES / "preds_seed", "seq": RES / "preds_seq",
         "con": RES / "preds_con"}
HLA = {"ELAGIGILTV": "A*02:01", "NLVPMVATV": "A*02:01",
       "GILGFVFTL": "A*02:01", "GLCTLVAML": "A*02:01",
       "RAKFKQLL": "B*08:01", "KLGGALQAK": "A*03:01",
       "AVFDRKSDAK": "A*11:01", "IVTDFSVIK": "A*11:01"}

def fused_scores(ds):
    """index -> fused score, and index -> fold. Rank-normalised per fold."""
    by_fold = defaultdict(lambda: defaultdict(dict))      # fold -> key -> idx -> score
    label, foldof = {}, {}
    for tree, root in TREES.items():
        for f in glob.glob(str(root / "init*" / f"*__{ds}_fold*.csv")):
            base = os.path.basename(f)[:-4]
            meth, rest = base.split("__", 1)
            k = int(rest.rsplit("_fold", 1)[1])
            init = os.path.basename(os.path.dirname(f))
            key = f"{meth}|{init}"
            for r in csv.DictReader(open(f)):
                i = str(int(float(r["index"])))
                by_fold[k][key][i] = float(r["pred"])
                label[i] = int(float(r["label"])); foldof[i] = k
    score = {}
    for k, members in by_fold.items():
        acc = defaultdict(list)
        for key, d in members.items():
            idx = list(d); vals = np.array([d[i] for i in idx])
            order = vals.argsort().argsort().astype(float)
            if len(order) > 1: order /= (len(order) - 1)
            for i, v in zip(idx, order): acc[i].append(v)
        for i, vs in acc.items(): score[i] = float(np.mean(vs))
    return score, label, foldof, len(by_fold[min(by_fold)])

def main():
    datasets = sorted(os.path.basename(p)[:-4] for p in glob.glob(str(SEQ / "*.csv")))
    rows = {ds: load_dataset(ds)[0] for ds in datasets}
    pos_of = {ds: {(r["fa"], r["fb"]) for r in rows[ds] if r["y"] == 1}
              for ds in datasets}

    out = {}
    for ds in datasets:
        score, label, foldof, nmem = fused_scores(ds)
        same, diff, unknown = [], [], []
        src = defaultdict(int)
        for r in rows[ds]:
            if r["y"] != 0: continue
            key = (r["fa"], r["fb"])
            owners = [o for o in datasets if o != ds and key in pos_of[o]]
            if not owners:
                unknown.append(r); continue
            for o in owners: src[o] += 1
            if any(HLA[o] == HLA[ds] for o in owners): same.append(r)
            else: diff.append(r)
        pos = [r for r in rows[ds] if r["y"] == 1]

        def sub_auc(negs):
            ii = [r["index"] for r in pos + negs if r["index"] in score]
            if not negs: return float("nan")
            y = [1] * len([r for r in pos if r["index"] in score]) + \
                [0] * len([r for r in negs if r["index"] in score])
            s = [score[i] for i in ii]
            return float(auc(np.array(y), np.array(s)))

        nneg = len(same) + len(diff) + len(unknown)
        out[ds] = dict(
            hla=HLA[ds], n_pos=len(pos), n_neg=nneg, n_members=nmem,
            neg_same_hla=len(same), neg_diff_hla=len(diff),
            neg_unknown=len(unknown),
            auc_all=sub_auc(same + diff + unknown),
            auc_vs_same_hla=sub_auc(same),
            auc_vs_diff_hla=sub_auc(diff),
            auc_vs_unknown=sub_auc(unknown),
            negatives_from=dict(sorted(src.items(), key=lambda kv: -kv[1])))
        o = out[ds]
        print(f"{ds:11s} {HLA[ds]:8s} pos={len(pos):4d} neg={nneg:4d} "
              f"(same-HLA {len(same):4d} / diff {len(diff):4d} / unknown {len(unknown):4d})  "
              f"AUC all={o['auc_all']:.3f}  vs-same={o['auc_vs_same_hla']:.3f}  "
              f"vs-diff={o['auc_vs_diff_hla']:.3f}  vs-unknown={o['auc_vs_unknown']:.3f}",
              flush=True)

    json.dump(out, open(RES / "neg_provenance.json", "w"), indent=1)
    print(f"\nwrote {RES / 'neg_provenance.json'}")

if __name__ == "__main__":
    main()
