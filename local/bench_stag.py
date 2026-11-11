#!/usr/bin/env python3
"""Time one training epoch per device at the real batch size (256) so the
full cross-validation runs on whichever device is actually faster here.
Usage: bench_stag.py <repo> <dataset> [epochs]"""
import sys, os, time
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
ds = sys.argv[2]
epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 3
sys.path.insert(0, str(repo))
os.chdir(repo)

import pandas as pd, numpy as np, torch, torch_geometric
import torch.nn as nn
from models.STAG.encode_structure import get_graph, add_edge_data
from models.STAG.model import STAG

df = pd.read_csv(repo / f"datasets/{ds}/{ds}.csv", index_col=0)
sdir = repo / f"datasets/{ds}/structures"

t0 = time.time()
graphs = []
for i, row in df.iterrows():
    g = get_graph(str(sdir / f"{int(i)}.pdb"), row.label, i)
    add_edge_data(g)
    graphs.append(g)
print(f"encoded {len(graphs)} graphs in {time.time()-t0:.1f}s")
nodes = sum(len(g.nodes) for g in graphs)
edges = sum(g.edge_index.shape[1] for g in graphs)
print(f"total nodes {nodes:,} edges {edges:,} (mean {nodes/len(graphs):.0f} / {edges/len(graphs):.0f} per graph)")

data_list = [torch_geometric.data.Data(x=g.get_node_data(), edge_index=g.edge_index,
                                       edge_attr=g.edge_attr,
                                       y=torch.tensor([g.label], dtype=torch.float32))
             for g in graphs]

for dname in ['cpu'] + (['mps'] if torch.backends.mps.is_available() else []):
    dev = torch.device(dname)
    loader = torch_geometric.loader.DataLoader(data_list, batch_size=256, shuffle=True)
    torch.manual_seed(0)
    model = STAG().to(dev)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    times = []
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        for batch in loader:
            batch = batch.to(dev)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = crit(out, batch.y.float().to(dev))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step(); opt.zero_grad()
        if dname == 'mps':
            torch.mps.synchronize()
        times.append(time.time() - t0)
    warm = np.mean(times[1:]) if len(times) > 1 else times[0]
    print(f"{dname}: epochs {[f'{t:.2f}s' for t in times]}  warm mean {warm:.2f}s/epoch"
          f"  -> 350 epochs x 5 folds = {warm*350*5/60:.0f} min upper bound")
