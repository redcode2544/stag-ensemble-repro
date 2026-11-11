#!/usr/bin/env python3
"""Paired-chain TCRs for OTHER epitopes -- the auxiliary negative pool.

The negatives in the STAG datasets are other peptides' 10x positives, so a
negative is only as trustworthy as that experiment. A TCR that a published
study showed binds a DIFFERENT epitope is a better negative: the evidence that
it is busy elsewhere is independent of the dextramer run that labelled our
target. This pulls those, excluding the 10x application note and our own 8
epitopes, and keeps one row per unique CDR3 pair.
"""
import csv, gzip, json, os, pathlib, sys
from collections import defaultdict

REL  = sys.argv[1]
HERE = pathlib.Path(__file__).resolve().parent.parent
OUT  = HERE / "ext"
OURS = {"ELAGIGILTV", "NLVPMVATV", "GILGFVFTL", "GLCTLVAML",
        "RAKFKQLL", "KLGGALQAK", "AVFDRKSDAK", "IVTDFSVIK"}
TENX = "10xgenomics.com"

def rdr(path):
    with gzip.open(path, "rt", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            yield r

rec = {}
for r in rdr(os.path.join(REL, "records.tsv.gz")):
    ep = r["antigen_epitope"]
    if r["species"] != "HomoSapiens" or ep in OURS: continue
    if TENX in r["reference_id"]: continue
    rec[r["record_id"]] = (ep, r["mhc_a"], r["reference_id"],
                           r["antigen_species"], r["vdjdb_score"])
print(f"candidate records: {len(rec)}", flush=True)

chains = defaultdict(dict)
for r in rdr(os.path.join(REL, "chains.tsv.gz")):
    if r["record_id"] in rec and r["gene"] in ("TRA", "TRB"):
        chains[r["record_id"]][r["gene"]] = (r["cdr3"], r["v_segm"])

seen, out, per_ep = set(), [], defaultdict(int)
for rid, (ep, mhc, ref, sp, sc) in rec.items():
    c = chains.get(rid, {})
    if "TRA" not in c or "TRB" not in c: continue
    a, va = c["TRA"]; b, vb = c["TRB"]
    if not a or not b or "*" in a or "*" in b: continue
    if (a, b) in seen: continue
    seen.add((a, b))
    out.append(dict(a=a, b=b, v_a=va, v_b=vb, ep=ep, mhc=mhc, ref=ref,
                    antigen=sp, score=sc))
    per_ep[ep] += 1

with open(OUT / "vdjdb_background.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["a", "b", "v_a", "v_b", "ep", "mhc",
                                        "ref", "antigen", "score"])
    w.writeheader(); w.writerows(out)
json.dump(dict(unique_pairs=len(out), epitopes=len(per_ep),
               top=dict(sorted(per_ep.items(), key=lambda kv: -kv[1])[:25])),
          open(OUT / "vdjdb_background_report.json", "w"), indent=1)
print(f"unique paired TCRs: {len(out)} over {len(per_ep)} epitopes")
for ep, n in sorted(per_ep.items(), key=lambda kv: -kv[1])[:12]:
    print(f"  {n:6d}  {ep}")
