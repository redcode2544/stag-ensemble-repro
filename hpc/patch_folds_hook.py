"""Add a STAG_FOLDS hook to the two baseline cross_validation files.

Idempotent, and it only replaces the one line that draws the fold vector, so
with the variable unset the files behave exactly as upstream does.
"""
import pathlib, sys
REPO = pathlib.Path('~/stag/STAG_public').expanduser()
OLD = "    df['fold'] =  np.random.randint(0,k,size=df.shape[0])"
NEW = """    import os as _os, json as _json
    _ff = _os.environ.get('STAG_FOLDS')
    if _ff and _os.path.exists(_ff):
        _m = {int(a): int(b) for a, b in _json.load(open(_ff)).items()}
        df['fold'] = [_m[int(_ix)] for _ix in df.index]
        print('folds read from ' + _ff)
    else:
        df['fold'] =  np.random.randint(0,k,size=df.shape[0])"""
n = 0
for m in ('Contacts', 'CNN_3D'):
    p = REPO / 'models' / m / 'cross_validation.py'
    s = p.read_text()
    if 'STAG_FOLDS' in s:
        print(f'{m}: already patched'); continue
    if OLD not in s:
        print(f'{m}: ANCHOR NOT FOUND - not patched'); sys.exit(1)
    p.write_text(s.replace(OLD, NEW, 1)); n += 1
    print(f'{m}: patched')
print(f'{n} file(s) changed')
