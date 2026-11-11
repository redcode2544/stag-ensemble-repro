"""Contacts / CNN_3D on a given k, reading the shared fold file."""
import warnings; warnings.filterwarnings('ignore')
import json, os, sys, time
import numpy as np, pandas as pd, torch

REPO = os.path.expanduser('~/stag/STAG_public')
method, name, k, tseed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
work = os.path.expanduser(f'~/stag/work/k{k}_{method}__{name}')
os.makedirs(os.path.join(work, 'output'), exist_ok=True)
os.chdir(work)
for link in ('datasets', 'models', 'evaluations.py'):
    if not os.path.exists(link):
        os.symlink(os.path.join(REPO, link), link)
sys.path.insert(0, work)

os.environ['STAG_FOLDS'] = os.path.expanduser(f'~/stag/folds_k{k}/{name}.json')
os.environ['STAG_DUMP'] = os.path.expanduser(f'~/stag/preds_k{k}/init{tseed}')
os.environ['STAG_TAG'] = f'{method}__{name}'
os.makedirs(os.environ['STAG_DUMP'], exist_ok=True)

mod = __import__(f'models.{method}.cross_validation', fromlist=['cross_validation'])
cv = mod.cross_validation
table, sdir = f'datasets/{name}/{name}.csv', f'datasets/{name}/structures'
df = pd.read_csv(table, index_col=0)
np.random.seed(20261107)
torch.manual_seed(tseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(tseed)

rec = dict(method=method, dataset=name, k=k, torch_seed=tseed, split_seed=20261107,
           complexes=int(df.shape[0]), positive_rate=float(df['label'].mean()),
           upstream_commit='06c30a57',
           deviation='STAG_FOLDS hook so every method shares one split')
t0 = time.time()
try:
    a = np.array(cv(table, sdir, k=k), dtype=float).ravel()
    rec.update(fold_auc=a.tolist(), mean_auc=float(a.mean()),
               sd_auc=float(a.std(ddof=1)) if len(a) > 1 else 0.0,
               folds_done=len(a), status='ok')
    print(f'{method} {name} k={k}: mean {a.mean():.4f}', flush=True)
except Exception as e:
    rec.update(status='failed', error=f'{type(e).__name__}: {str(e)[:400]}')
    print(f'{method} {name} k={k}: FAILED {type(e).__name__}: {e}', flush=True)
rec['seconds'] = round(time.time() - t0, 1)
out = os.path.expanduser(f'~/stag/results_k{k}/{method}__{name}__init{tseed}.json')
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(rec, open(out, 'w'), indent=2)
print(f'-> {out} ({rec["seconds"]/60:.1f} min)', flush=True)
