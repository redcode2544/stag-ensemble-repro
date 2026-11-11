#!/usr/bin/env python3
"""Smoke test: encode a few STAG structures and run one forward/backward pass
on each available device. Usage: smoke_stag.py <repo> <dataset-name> [n]"""
import sys, os, time
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
ds = sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
sys.path.insert(0, str(repo))
os.chdir(repo)

import pandas as pd, numpy as np, torch, torch_geometric
import torch.nn as nn

from models.STAG.encode_structure import get_graph, add_edge_data
from models.STAG.model import STAG

table = repo / f"datasets/{ds}/{ds}.csv"
sdir = repo / f"datasets/{ds}/structures"
df = pd.read_csv(table, index_col=0)
print(f"{ds}: {df.shape[0]} rows, label mean {df['label'].mean():.3f}")

t0 = time.time()
graphs = []
for i, row in df.head(n).iterrows():
    g = get_graph(str(sdir / f"{int(i)}.pdb"), row.label, i)
    add_edge_data(g)
    g.index = i
    graphs.append(g)
enc = time.time() - t0
print(f"encoded {len(graphs)} graphs in {enc:.1f}s ({enc/len(graphs):.2f}s each)")
g0 = graphs[0]
print(f"  graph0: nodes={len(g0.nodes)} node_feat={tuple(g0.get_node_data().shape)} "
      f"edges={g0.edge_index.shape[1]} edge_feat={tuple(g0.edge_attr.shape)} label={g0.label}")

data_list = [torch_geometric.data.Data(x=g.get_node_data(), edge_index=g.edge_index,
                                       edge_attr=g.edge_attr,
                                       y=torch.tensor([g.label], dtype=torch.float32))
             for g in graphs]
loader = torch_geometric.loader.DataLoader(data_list, batch_size=len(data_list), shuffle=False)

devs = ['cpu'] + (['mps'] if torch.backends.mps.is_available() else [])
for dname in devs:
    dev = torch.device(dname)
    torch.manual_seed(0)
    model = STAG().to(dev)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    try:
        t0 = time.time()
        for batch in loader:
            batch = batch.to(dev)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = crit(out, batch.y.float().to(dev))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step(); opt.zero_grad()
        gnorm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None) if False else None
        print(f"  {dname}: OK  out={out.detach().cpu().numpy().round(4).tolist()}  "
              f"loss={loss.item():.4f}  {time.time()-t0:.2f}s")
    except Exception as e:
        print(f"  {dname}: FAILED  {type(e).__name__}: {e}")

print(f"params: {sum(p.numel() for p in STAG().parameters()):,}")
