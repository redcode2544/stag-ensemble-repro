"""Cross-validation for the three graph encoders, any k, one protocol.

The training protocol is copied from models/STAG/cross_validation.py line for
line -- 350 epochs, Adam at 1e-4, BCEWithLogits with the positive class
weighted by the class ratio, batch 256 train / 64 test, early stopping with
patience 25 after epoch 50, and the predictions dumped from the best epoch.
That protocol selects its stopping epoch ON THE TEST FOLD, which inflates every
number it produces; it is kept unchanged anyway, because changing it would make
these numbers incomparable with the published ones, and the inflation applies
equally to every method being compared.

Differences from upstream, all deliberate:
  * k is a real parameter and the fold assignment is taken from a dict instead
    of `df.loc[graph.fold]['fold']`, which returns a Series and crashes when the
    CSV has duplicate indices (upstream's pan_peptide table does).
  * the fold vector can be loaded from a file so that every method and every
    initialisation sees exactly the same folds.
  * Data objects carry `pos` (residue coordinates) and `is_pep` (peptide-chain
    mask), which the two new encoders need and the original does not read.
"""
import copy, json, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch_geometric
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

from models.STAG.cross_validation import get_graphs, init_weights
from models.STAG.model import STAG
from .model2 import STAG_EGNN, STAG_DUAL, semantic_loss

EPOCHS, PATIENCE, MIN_EPOCH = 350, 25, 50
LR, BATCH_TRAIN, BATCH_TEST = 1e-4, 256, 64
SEM_WEIGHT = 0.1


def to_data(graph):
    coords = graph.get_coords()
    is_pep = torch.tensor([n.chain == 'P' for n in graph.nodes], dtype=torch.bool)
    return torch_geometric.data.Data(
        x=graph.get_node_data(), edge_index=graph.edge_index,
        edge_attr=graph.edge_attr, pos=coords, is_pep=is_pep,
        y=int(graph.label))


def build(name):
    if name == 'STAG':      return STAG(), False
    if name == 'STAG_EGNN': return STAG_EGNN(), False
    if name == 'STAG_DUAL': return STAG_DUAL(), True
    raise ValueError(name)


def forward(model, data, name):
    if name == 'STAG':
        return model(data.x, data.edge_index, data.edge_attr, data.batch), None
    return model(data)


def run(loader, model, name, criterion, optimizer, dual, training=False):
    model.train() if training else model.eval()
    y_truth, y_hat = [], []
    for data in loader:
        data = data.to('cuda')
        out, sem = forward(model, data, name)
        if out.ndim == 0:
            out = out.unsqueeze(0)
        if training:
            loss = criterion(out, data.y.float().to('cuda'))
            if dual and sem is not None:
                loss = loss + SEM_WEIGHT * semantic_loss(sem, data.y.to('cuda'))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            optimizer.zero_grad()
        y_truth = np.concatenate((y_truth, data.y.detach().to('cpu').numpy()))
        y_hat = np.concatenate((y_hat, out.detach().to('cpu').numpy()))
    return roc_auc_score(y_truth, y_hat)


def cross_validation(data_table, data_dir, method='STAG', k=5, fold_file=None,
                     dump=None, tag=None, verbose=False, graphs=None,
                     val_frac=0.0):
    """val_frac > 0 switches early stopping from the TEST fold to a validation
    split carved out of the training folds.

    Upstream stops on the test fold and reports the best test AUC it saw, which
    selects the epoch on the thing being measured. That inflates every number
    it produces, and it inflates MORE when the test fold is small -- so moving
    from 5 folds to 10 raises the score partly by giving the selector a noisier
    statistic to maximise. Running the same thing both ways is the only way to
    tell that apart from the genuine gain of training on 90% instead of 80%.
    """
    if graphs is None:
        print(f'encoding structures as graphs ({method}, k={k})', flush=True)
        graphs = get_graphs(data_table, data_dir)
    df = pd.read_csv(data_table, index_col=0)

    if fold_file and os.path.exists(fold_file):
        fold_of = {int(a): int(b) for a, b in json.load(open(fold_file)).items()}
        print(f'folds read from {fold_file}', flush=True)
    else:
        fold_of = {int(i): int(f) for i, f in
                   zip(df.index, np.random.randint(0, k, size=df.shape[0]))}
        if fold_file:
            os.makedirs(os.path.dirname(fold_file), exist_ok=True)
            json.dump({str(a): b for a, b in fold_of.items()}, open(fold_file, 'w'))
            print(f'folds written to {fold_file}', flush=True)

    pos_rate = float(np.mean(df['label']))
    AUCs, t0 = [], time.time()
    for i in range(k):
        data = copy.deepcopy(graphs)
        tr = [g for g in data if fold_of[int(g.fold)] != i]
        te = [g for g in data if fold_of[int(g.fold)] == i]
        if not te or len({int(g.label) for g in te}) < 2:
            print(f'fold {i}: skipped ({len(te)} test graphs)', flush=True)
            continue
        ids = np.array([g.index for g in te])
        val = []
        if val_frac > 0:
            rng = np.random.RandomState(1000 + i)
            by_lab = {0: [], 1: []}
            for gi, g in enumerate(tr):
                by_lab[int(g.label)].append(gi)
            pick = set()
            for lab, idxs in by_lab.items():
                m = max(1, int(round(val_frac * len(idxs))))
                pick |= set(rng.choice(idxs, size=min(m, len(idxs)), replace=False))
            val = [g for gi, g in enumerate(tr) if gi in pick]
            tr = [g for gi, g in enumerate(tr) if gi not in pick]
        train_loader = torch_geometric.loader.DataLoader(
            [to_data(g) for g in tr], batch_size=BATCH_TRAIN, shuffle=True)
        test_loader = torch_geometric.loader.DataLoader(
            [to_data(g) for g in te], batch_size=BATCH_TEST, shuffle=False)
        val_loader = (torch_geometric.loader.DataLoader(
            [to_data(g) for g in val], batch_size=BATCH_TEST, shuffle=False)
            if val else None)

        model, dual = build(method)
        model = model.to('cuda')
        model.apply(init_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor((1 - pos_rate) / pos_rate).to('cuda'))

        best, stop, best_rows, reported = 0.0, 0, None, 0.0
        with tqdm(range(EPOCHS), disable=not verbose) as t:
            for epoch in t:
                run(train_loader, model, method, criterion, optimizer, dual, True)
                with torch.no_grad():
                    test_auc = run(test_loader, model, method, criterion,
                                   optimizer, dual)
                    watch = (run(val_loader, model, method, criterion, optimizer,
                                 dual) if val_loader is not None else test_auc)
                if watch > best:
                    stop, best = 0, watch
                    reported = test_auc
                    with torch.no_grad():
                        model.eval()
                        yy, pp = [], []
                        for d in test_loader:
                            d = d.to('cuda')
                            o, _ = forward(model, d, method)
                            if o.ndim == 0:
                                o = o.unsqueeze(0)
                            yy = np.concatenate((yy, d.y.detach().to('cpu').numpy()))
                            pp = np.concatenate((pp, o.detach().to('cpu').numpy()))
                        best_rows = pd.DataFrame(
                            {'index': ids[:len(pp)], 'label': yy, 'pred': pp})
                stop += 1
                if stop > PATIENCE and epoch > MIN_EPOCH:
                    break
        AUCs.append(reported if val_loader is not None else best)
        print(f'fold {i}: {len(tr)} train / {len(val)} val / {len(te)} test, '
              f'test AUC {AUCs[-1]:.4f} (watch {best:.4f}) at epoch '
              f'{epoch - stop + 1}', flush=True)
        if dump and best_rows is not None:
            os.makedirs(dump, exist_ok=True)
            best_rows = best_rows.copy()
            best_rows['fold'] = i
            best_rows.to_csv(f'{dump}/{tag}_fold{i}.csv', index=False)
        for obj in (data, tr, te, val, train_loader, test_loader, val_loader,
                    model, optimizer, criterion):
            del obj
        torch.cuda.empty_cache()

    print('#' * 50)
    print(f'{method} k={k}: ROC-AUCs {[round(a, 4) for a in AUCs]} '
          f'mean {np.mean(AUCs):.4f} in {(time.time() - t0) / 60:.1f} min', flush=True)
    return AUCs
