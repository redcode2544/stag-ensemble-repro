#!/usr/bin/env python3
"""Merge three curated TCR databases into one external positive set per epitope.

Sources, all paired alpha+beta, all in the IMGT junction convention (C...F)
that the sequences extracted from the structure files use:

  VDJdb  2026-11-06 release (ext/vdjdb_<EP>.csv, written by vdjdb_pull.py)
  McPAS-TCR  full database CSV
  IEDB   tcr_full_v3.csv (receptor export)

The 10x application note that produced every STAG label is excluded from all
three, so what is left is independent evidence:
  * VDJdb tags it in reference_id (10xgenomics.com);
  * McPAS does not contain it at all (every row carries a PubMed ID);
  * IEDB curates it as reference 1036521 -- identified here without trusting a
    name: it is the ONLY source IEDB has for KLGGALQAK, and it has by far the
    largest overlap with the STAG datasets' own TCRs for every epitope.

Writes ext/curated_<EP>.csv (a,b,src,ref,mhc) and ext/curated_report.json.
"""
import csv, collections, io, json, os, pathlib, sys

csv.field_size_limit(10 ** 7)
HERE = pathlib.Path(__file__).resolve().parent.parent
EXT  = HERE / "ext"
EPIS = ["ELAGIGILTV", "NLVPMVATV", "GILGFVFTL", "GLCTLVAML",
        "RAKFKQLL", "KLGGALQAK", "AVFDRKSDAK", "IVTDFSVIK"]
TENX_VDJDB = "10xgenomics.com"
TENX_IEDB  = "1036521"

mcpas_path = sys.argv[1]
iedb_path  = sys.argv[2]

out = {ep: {} for ep in EPIS}          # (a,b) -> row, first source wins
cnt = collections.defaultdict(lambda: collections.Counter())

def add(ep, a, b, src, ref, mhc):
    if not a or not b: return
    if not a.startswith("C") or not b.startswith("C"): return
    cnt[ep][src + "_raw"] += 1
    if (a, b) in out[ep]: return
    out[ep][(a, b)] = dict(a=a, b=b, src=src, ref=ref, mhc=mhc)
    cnt[ep][src] += 1

# ---- VDJdb (already filtered to the 8 epitopes by vdjdb_pull.py) ---------
for ep in EPIS:
    for r in csv.DictReader(open(EXT / f"vdjdb_{ep}.csv")):
        if TENX_VDJDB in r["ref"]: continue
        add(ep, r["a"], r["b"], "vdjdb", r["ref"], r["mhc"])

# ---- McPAS --------------------------------------------------------------
for r in csv.DictReader(io.open(mcpas_path, encoding="utf-8-sig")):
    ep = (r.get("Epitope.peptide") or "").strip()
    if ep not in out: continue
    add(ep, (r.get("CDR3.alpha.aa") or "").strip(),
        (r.get("CDR3.beta.aa") or "").strip(), "mcpas",
        "PMID:" + (r.get("PubMed.ID") or "?"), (r.get("MHC") or "").strip())

# ---- IEDB ---------------------------------------------------------------
fh = open(iedb_path, newline="", encoding="utf-8", errors="replace")
rd = csv.reader(fh); next(rd); next(rd)
REF, EP, MHC, T1, J1, T2, J2 = 4, 6, 11, 12, 25, 43, 56
for row in rd:
    if len(row) <= J2: continue
    ep = row[EP].strip()
    if ep not in out: continue
    if TENX_IEDB in row[REF]: continue
    if row[T1] != "alpha" or row[T2] != "beta": continue
    add(ep, row[J1].strip(), row[J2].strip(), "iedb", row[REF], row[MHC])

report = {}
for ep in EPIS:
    with open(EXT / f"curated_{ep}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["a", "b", "src", "ref", "mhc"])
        w.writeheader(); w.writerows(out[ep].values())
    c = cnt[ep]
    report[ep] = dict(unique=len(out[ep]),
                      new_from={s: c[s] for s in ("vdjdb", "mcpas", "iedb")},
                      rows_seen={s: c[s + "_raw"] for s in ("vdjdb", "mcpas", "iedb")},
                      studies=len({r["ref"] for r in out[ep].values()}))
    print(f"{ep:11s} unique {len(out[ep]):6d}  "
          f"(vdjdb {c['vdjdb']:5d} + mcpas {c['mcpas']:5d} + iedb {c['iedb']:5d} new)  "
          f"studies {report[ep]['studies']}")
json.dump(report, open(EXT / "curated_report.json", "w"), indent=1)
print(f"\nwrote {EXT}/curated_<EPITOPE>.csv + curated_report.json")
