"""Write one fold vector per (dataset, k), once, so every method and every
initialisation is scored on exactly the same split.

The draw is `np.random.randint(0, k, size=n)` after `np.random.seed(20261107)` --
the same call and the same seed the 5-fold runs used, so the k=5 files
reproduce the folds already on disk rather than inventing a new split.
"""
import json, os, sys
import numpy as np, pandas as pd

REPO = os.path.expanduser('~/stag/STAG_public')
SPLIT_SEED = 20261107
k = int(sys.argv[1])
names = sorted(d for d in os.listdir(f'{REPO}/datasets')
               if os.path.isdir(f'{REPO}/datasets/{d}') and d != 'pan_peptide')
out = os.path.expanduser(f'~/stag/folds_k{k}')
os.makedirs(out, exist_ok=True)
for n in names:
    df = pd.read_csv(f'{REPO}/datasets/{n}/{n}.csv', index_col=0)
    np.random.seed(SPLIT_SEED)
    f = np.random.randint(0, k, size=df.shape[0])
    json.dump({str(int(i)): int(v) for i, v in zip(df.index, f)},
              open(f'{out}/{n}.json', 'w'))
    sizes = np.bincount(f, minlength=k)
    print(f'{n:12s} n={df.shape[0]:5d} folds={sizes.tolist()} '
          f'pos={df["label"].mean():.3f}')
print(f'wrote {out}')
