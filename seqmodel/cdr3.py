#!/usr/bin/env python3
"""Locate CDR3 inside a modelled TCR variable domain.

IMGT junction convention: CDR3 runs from the conserved Cys (104) through the
conserved Phe/Trp of the F/W-G-X-G motif (118). We search for that bracket and
take the LAST qualifying match, because the framework Cys we want is the one
closest to the J-region motif.
"""
import re

JUNCTION = re.compile(r"C(?=([A-Z]{4,30}?[FW]G[A-Z]G))")

def cdr3(chain):
    """-> (cdr3_with_anchors, start_index) or (None, -1) when no motif is found."""
    best = None
    for m in JUNCTION.finditer(chain):
        inner = m.group(1)
        end = m.end() + len(inner)          # past the G of F/W-G-X-G
        # IMGT junction = C ... F/W  (drop the trailing G-X-G)
        junc = chain[m.start():end - 3]
        if 6 <= len(junc) <= 28:
            best = (junc, m.start())
    return best if best else (None, -1)

if __name__ == "__main__":
    import csv, glob, os, collections
    for f in sorted(glob.glob("seq/*.csv")):
        ds = os.path.basename(f)[:-4]
        rows = list(csv.DictReader(open(f)))
        stat = {}
        for ch in ("tcra", "tcrb"):
            got = [cdr3(r[ch])[0] for r in rows]
            ok = [g for g in got if g]
            lens = collections.Counter(len(g) for g in ok)
            starts = collections.Counter(g[:3] for g in ok).most_common(3)
            stat[ch] = (len(ok), len(rows), sorted(lens)[0] if lens else 0,
                        sorted(lens)[-1] if lens else 0, starts)
        print(f"{ds:12s}")
        for ch in ("tcra", "tcrb"):
            n, tot, lo, hi, starts = stat[ch]
            tag = "OK " if n == tot else "GAP"
            print(f"   {tag} {ch}: {n}/{tot} found · len {lo}-{hi} · "
                  f"commonest starts {[f'{s}({c})' for s, c in starts]}")
