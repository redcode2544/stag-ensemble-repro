"""One (method, dataset, k, torch-seed) run. Submitted by sbatch, never run on
the login node.

The fold vector is written once per (dataset, k) into ~/stag/folds_k<k>/ and
read by every later run, so every method and every initialisation is scored on
exactly the same folds -- the condition for averaging their predictions
without leaking training data into the test fold.
"""
import warnings; warnings.filterwarnings('ignore')
import json, os, sys, time
import numpy as np, pandas as pd, torch

sys.path.insert(0, '.')
from models.STAG2.cv2 import cross_validation as cv2

method, name, k, tseed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
SPLIT_SEED = 20261107                      # held constant: fixes the fold draw

table = f'datasets/{name}/{name}.csv'
sdir = f'datasets/{name}/structures'
df = pd.read_csv(table, index_col=0)
folds = os.path.expanduser(f'~/stag/folds_k{k}/{name}.json')
dump = os.path.expanduser(f'~/stag/preds_k{k}/init{tseed}')

np.random.seed(SPLIT_SEED)
torch.manual_seed(tseed); torch.cuda.manual_seed_all(tseed)

rec = dict(method=method, dataset=name, k=k, torch_seed=tseed,
           split_seed=SPLIT_SEED, complexes=int(df.shape[0]),
           positive_rate=float(df['label'].mean()),
           gpu=torch.cuda.get_device_name(0), upstream_commit='06c30a57')
t0 = time.time()
try:
    a = np.array(cv2(table, sdir, method=method, k=k, fold_file=folds,
                     dump=dump, tag=f'{method}__{name}'), dtype=float)
    rec.update(fold_auc=[float(x) for x in a], mean_auc=float(a.mean()),
               sd_auc=float(a.std(ddof=1)) if len(a) > 1 else 0.0,
               folds_done=len(a), status='ok')
except Exception as e:
    rec.update(status='failed', error=f'{type(e).__name__}: {str(e)[:400]}')
    print(f'FAILED {type(e).__name__}: {e}', flush=True)
rec['seconds'] = round(time.time() - t0, 1)
rec['peak_gpu_gib'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
out = os.path.expanduser(f'~/stag/results_k{k}/{method}__{name}__init{tseed}.json')
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(rec, open(out, 'w'), indent=2)
print(f'{method} {name} k={k} init{tseed}: {rec.get("mean_auc", float("nan")):.4f} '
      f'in {rec["seconds"]/60:.1f} min -> {out}', flush=True)
