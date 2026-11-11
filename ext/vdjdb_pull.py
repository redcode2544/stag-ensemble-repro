#!/usr/bin/env python3
"""Pull curated paired-chain TCRs for STAG's 8 epitopes out of a VDJdb release.

Why: every label in the STAG datasets comes from one 10x dextramer experiment.
A second, independent source of positives is the only thing four different
model families cannot substitute for. VDJdb aggregates published tetramer /
dextramer / functional studies and records WHICH study each row came from, so
the 10x rows can be excluded and what remains is genuinely other evidence.

Writes ext/vdjdb_<EPITOPE>.csv (a,b,v_a,v_b,ref,method,score,mhc) and
ext/vdjdb_report.json with the per-epitope, per-study counts.
"""
import csv, gzip, json, os, pathlib, sys
from collections import defaultdict

REL = sys.argv[1] if len(sys.argv) > 1 else ""
HERE = pathlib.Path(__file__).resolve().parent.parent
OUT  = HERE / "ext"
EPIS = ["ELAGIGILTV", "NLVPMVATV", "GILGFVFTL", "GLCTLVAML",
        "RAKFKQLL", "KLGGALQAK", "AVFDRKSDAK", "IVTDFSVIK"]

def rdr(path):
    with gzip.open(path, "rt", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            yield r

want = set(EPIS)
rec = {}
for r in rdr(os.path.join(REL, "records.tsv.gz")):
    if r["species"] != "HomoSapiens" or r["antigen_epitope"] not in want:
        continue
    rec[r["record_id"]] = dict(
        ep=r["antigen_epitope"], mhc=r["mhc_a"], ref=r["reference_id"],
        method=r["method_identification"], single=r["method_singlecell"],
        score=r["vdjdb_score"], subject=r["meta_subject_id"],
        study=r["meta_study_id"], gene=r["antigen_gene"],
        sp=r["antigen_species"])
print(f"records matching the 8 epitopes: {len(rec)}", flush=True)

chains = defaultdict(dict)
for r in rdr(os.path.join(REL, "chains.tsv.gz")):
    if r["record_id"] not in rec:
        continue
    g = r["gene"]
    if g in ("TRA", "TRB"):
        chains[r["record_id"]][g] = (r["cdr3"], r["v_segm"], r["j_segm"])

rows = defaultdict(list)
per_ref = defaultdict(lambda: defaultdict(int))
unpaired = defaultdict(int)
for rid, m in rec.items():
    c = chains.get(rid, {})
    if "TRA" not in c or "TRB" not in c:
        unpaired[m["ep"]] += 1
        continue
    a, va, ja = c["TRA"]; b, vb, jb = c["TRB"]
    if not a or not b or "*" in a or "*" in b:
        unpaired[m["ep"]] += 1
        continue
    rows[m["ep"]].append(dict(a=a, b=b, v_a=va, j_a=ja, v_b=vb, j_b=jb,
                              ref=m["ref"], method=m["method"],
                              score=m["score"], mhc=m["mhc"],
                              subject=m["subject"], record=rid))
    per_ref[m["ep"]][m["ref"]] += 1

report = {}
OUT.mkdir(exist_ok=True)
for ep in EPIS:
    rs = rows[ep]
    uniq = {(r["a"], r["b"]): r for r in rs}
    with open(OUT / f"vdjdb_{ep}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["a", "b", "v_a", "j_a", "v_b", "j_b",
                                           "ref", "method", "score", "mhc",
                                           "subject", "record"])
        w.writeheader()
        for r in uniq.values():
            w.writerow(r)
    report[ep] = dict(paired=len(rs), unique_pairs=len(uniq),
                      unpaired_or_bad=unpaired[ep],
                      by_reference=dict(sorted(per_ref[ep].items(),
                                               key=lambda kv: -kv[1])))
    print(f"{ep:11s} paired={len(rs):6d} unique={len(uniq):6d} "
          f"unpaired={unpaired[ep]:6d} studies={len(per_ref[ep])}", flush=True)

json.dump(report, open(OUT / "vdjdb_report.json", "w"), indent=1)
print(f"\nwrote {OUT}/vdjdb_<EPITOPE>.csv + vdjdb_report.json")
