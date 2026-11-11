"""All graph methods for ONE dataset in one process.

Encoding the structures into graphs costs about a quarter of a second per
complex and does not depend on the method or the initialisation, so doing it
once per dataset instead of once per run saves roughly two thirds of the GPU
time on the big datasets.

usage: run_dataset.py NAME K INITS [--methods A,B] [--val FRAC] [--suffix S]
"""
import warnings; warnings.filterwarnings('ignore')
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch

sys.path.insert(0, '.')
from models.STAG.cross_validation import get_graphs
from models.STAG2.cv2 import cross_validation as cv2

ap = argparse.ArgumentParser()
ap.add_argument('name'); ap.add_argument('k', type=int); ap.add_argument('inits', type=int)
ap.add_argument('--methods', default='STAG,STAG_EGNN,STAG_DUAL')
ap.add_argument('--val', type=float, default=0.0)
ap.add_argument('--suffix', default='')
a = ap.parse_args()

SPLIT_SEED = 20261107
table, sdir = f'datasets/{a.name}/{a.name}.csv', f'datasets/{a.name}/structures'
df = pd.read_csv(table, index_col=0)
folds = os.path.expanduser(f'~/stag/folds_k{a.k}/{a.name}.json')
print(f'=== {a.name}: {df.shape[0]} complexes, {df["label"].mean()*100:.1f}% positive, '
      f'k={a.k}, inits={a.inits}, val={a.val}, methods={a.methods}', flush=True)

t0 = time.time()
graphs = get_graphs(table, sdir)
print(f'encoded {len(graphs)} graphs in {(time.time()-t0)/60:.1f} min', flush=True)

for method in a.methods.split(','):
    for init in range(1, a.inits + 1):
        np.random.seed(SPLIT_SEED)
        torch.manual_seed(init); torch.cuda.manual_seed_all(init)
        dump = os.path.expanduser(f'~/stag/preds_k{a.k}{a.suffix}/init{init}')
        rec = dict(method=method, dataset=a.name, k=a.k, torch_seed=init,
                   split_seed=SPLIT_SEED, val_frac=a.val,
                   complexes=int(df.shape[0]),
                   positive_rate=float(df['label'].mean()),
                   gpu=torch.cuda.get_device_name(0), upstream_commit='06c30a57')
        t1 = time.time()
        try:
            auc = np.array(cv2(table, sdir, method=method, k=a.k, fold_file=folds,
                               dump=dump, tag=f'{method}__{a.name}', graphs=graphs,
                               val_frac=a.val), dtype=float)
            rec.update(fold_auc=[float(x) for x in auc], mean_auc=float(auc.mean()),
                       sd_auc=float(auc.std(ddof=1)) if len(auc) > 1 else 0.0,
                       folds_done=len(auc), status='ok')
        except Exception as e:
            rec.update(status='failed', error=f'{type(e).__name__}: {str(e)[:400]}')
            print(f'FAILED {method} {a.name} init{init}: {type(e).__name__}: {e}', flush=True)
        rec['seconds'] = round(time.time() - t1, 1)
        out = os.path.expanduser(
            f'~/stag/results_k{a.k}{a.suffix}/{method}__{a.name}__init{init}.json')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(rec, open(out, 'w'), indent=2)
        print(f'>>> {method} {a.name} k={a.k} init{init}: '
              f'{rec.get("mean_auc", float("nan")):.4f} in {rec["seconds"]/60:.1f} min',
              flush=True)
print(f'=== {a.name} done, total {(time.time()-t0)/60:.1f} min', flush=True)
