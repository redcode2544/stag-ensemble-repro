#!/usr/bin/env python3
"""Which curated studies are actually INDEPENDENT of the STAG labels?

Excluding the 10x application note by name is not enough. A later paper can
re-report the same experiment's clonotypes under its own reference, and the
curated databases will file them under that paper. Reading every paper does not
scale and would not be checkable; this tests it from the data instead.

For each study S and dataset d: what fraction of S's OWN alpha+beta pairs for
this epitope appear EXACTLY in d? Public clonotypes share the beta motif and
rarely the exact pair, so a high exact-pair rate means shared provenance, not
independent confirmation.

Found this way: VDJdb files 181 NLVPMVATV pairs under a 2020 medRxiv preprint,
and 159 of them (88%) are exact alpha+beta duplicates of this dataset's
positives, while sharing zero pairs with VDJdb's own curation of the 10x note.
Treating those as outside evidence would have inflated NLVPMVATV by ~0.3 AUC.

Writes ext/curated_<EP>_indep.csv (the studies that pass) + the audit JSON.
"""
import csv, json, pathlib, sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "seqmodel"))
from cdr3 import cdr3                                        # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
EXT  = HERE / "ext"
SEQ  = HERE / "seq"
TENX = "10xgenomics.com"
RATE, MINN = 0.20, 5          # flag a study at >=20% exact-pair reuse, >=5 pairs

def own_pairs(ds):
    pairs, pos = set(), set()
    for r in csv.DictReader(open(SEQ / f"{ds}.csv")):
        a, b = cdr3(r["tcra"])[0], cdr3(r["tcrb"])[0]
        if not a or not b: continue
        pairs.add((a, b))
        if int(float(r["label"])) == 1: pos.add((a, b))
    return pairs, pos

audit, kept_all = {}, {}
for f in sorted(EXT.glob("curated_*.csv")):
    if f.name.endswith("_indep.csv"): continue
    ds = f.stem[len("curated_"):]
    pairs, pos = own_pairs(ds)
    by = defaultdict(list)
    for r in csv.DictReader(open(f)):
        if TENX in r["ref"]: continue
        by[r["ref"]].append(r)
    rows, flagged = [], []
    for ref, rs in by.items():
        p = {(r["a"], r["b"]) for r in rs}
        hit = len(p & pairs); hpos = len(p & pos)
        rate = hit / len(p)
        rec = dict(ref=ref, pairs=len(p), exact_in_dataset=hit,
                   exact_in_positives=hpos, rate=rate,
                   dropped=bool(rate >= RATE and len(p) >= MINN))
        if rec["dropped"]:
            flagged.append(rec)
        else:
            rows.extend(rs)
        audit.setdefault(ds, []).append(rec)
    with open(EXT / f"curated_{ds}_indep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["a", "b", "src", "ref", "mhc"])
        w.writeheader(); w.writerows(rows)
    kept_all[ds] = len(rows)
    tot = sum(len(v) for v in by.values())
    print(f"{ds:11s} studies {len(by):3d} -> kept {len(by) - len(flagged):3d} | "
          f"rows {tot:6d} -> {len(rows):6d}"
          + ("  DROPPED: " + ", ".join(f"{r['ref'][:46]} ({r['pairs']} pairs, "
                                       f"{r['rate']:.0%} exact)" for r in flagged)
             if flagged else ""))

json.dump(dict(rule=dict(exact_pair_rate=RATE, min_pairs=MINN), audit=audit,
               kept=kept_all), open(EXT / "provenance_audit.json", "w"), indent=1)
print(f"\nwrote {EXT}/curated_<EP>_indep.csv + provenance_audit.json")
