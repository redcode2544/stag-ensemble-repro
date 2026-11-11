#!/usr/bin/env python3
"""Embed the TCR V-domains with ESM-2 and cache the vectors.

The four existing streams all describe the receptor with hand-made features --
atom graphs, contact counts, voxel grids, character n-grams. A protein language
model brings something none of them has: statistics learned from millions of
unrelated protein sequences. That is outside knowledge about protein sequence
in general, which is why it is worth a try even though "another featurisation
of a source already in the pool" has not paid before in this project.

Embeds each unique chain once (mean over residue representations of the last
layer) and caches to runs/results/esm/<model>.npz keyed by the sequence itself,
so re-running costs nothing.
"""
import argparse, csv, glob, os, pathlib, sys
import numpy as np
import torch, esm

HERE = pathlib.Path(__file__).resolve().parent.parent
SEQ  = HERE / "seq"
OUT  = HERE / "runs" / "results" / "esm"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esm2_t12_35M_UR50D")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    seqs = set()
    for f in sorted(glob.glob(str(SEQ / "*.csv"))):
        for r in csv.DictReader(open(f)):
            seqs.add(r["tcra"]); seqs.add(r["tcrb"])
    seqs = sorted(s for s in seqs if s)
    print(f"{len(seqs)} unique chains to embed", flush=True)

    model, alphabet = getattr(esm.pretrained, args.model)()
    model.eval().to(args.device)
    bc = alphabet.get_batch_converter()
    layer = model.num_layers

    vecs = np.zeros((len(seqs), model.embed_dim), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(seqs), args.batch):
            chunk = seqs[i:i + args.batch]
            _, _, toks = bc([(str(j), s) for j, s in enumerate(chunk)])
            toks = toks.to(args.device)
            rep = model(toks, repr_layers=[layer])["representations"][layer]
            for j, s in enumerate(chunk):
                vecs[i + j] = rep[j, 1:len(s) + 1].mean(0).float().cpu().numpy()
            if (i // args.batch) % 20 == 0:
                print(f"  {i + len(chunk)}/{len(seqs)}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{args.model}.npz"
    np.savez_compressed(p, seqs=np.array(seqs), vecs=vecs)
    print(f"wrote {p}  shape {vecs.shape}")

if __name__ == "__main__":
    main()
