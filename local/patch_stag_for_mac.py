#!/usr/bin/env python3
"""Minimal, reversible edits so KavrakiLab/STAG_public runs on an Apple-Silicon Mac.

Scope: device handling, pandas 2.x API removal, and a torch-cluster-free
radius_graph fallback. The model architecture, hyper-parameters, features and
cross-validation protocol are left exactly as upstream wrote them.

Run:  python3 patch_stag_for_mac.py /path/to/STAG_public
Every change is a literal string replacement that fails loudly if the
upstream text is not found, so it can never half-apply silently.
"""
import sys
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()

DEVICE_HEADER = '''
# --- local: device selection (upstream hardcoded 'cuda') ---
import os as _os
def _pick_device():
    want = _os.environ.get('STAG_DEVICE', 'auto').lower()
    if want not in ('auto', 'cpu', 'mps', 'cuda'):
        raise ValueError("STAG_DEVICE must be auto, cpu, mps or cuda")
    if want != 'auto':
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

DEVICE = _pick_device()

def _empty_cache():
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    elif DEVICE.type == 'mps':
        torch.mps.empty_cache()
# --- end local ---
'''

# (file, old, new, description)
EDITS = [
    # ---- model.py: PyG >= 2.5 generates propagate() from the propagate_type
    # annotation, which upstream left stale (it lists x/edge_attr while the call
    # passes query/key/value/edge_attr). Older PyG accepted loose kwargs.
    ("models/STAG/model.py",
     "        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)\n",
     "        # propagate_type: (query: Tensor, key: Tensor, value: Tensor, edge_attr: OptTensor)\n",
     "fix stale propagate_type annotation for PyG >= 2.5"),

    # ---- cross_validation.py: device + pandas 2 ----
    ("models/STAG/cross_validation.py",
     "from .encode_structure import *\nfrom .model import *\nfrom .utils import *\n",
     "from .encode_structure import *\nfrom .model import *\nfrom .utils import *\n" + DEVICE_HEADER,
     "insert device selection helper"),

    ("models/STAG/cross_validation.py",
     "    for data in loader:\n        data = data.to('cuda')\n        out = model(data.x,data.edge_index,data.edge_attr,data.batch)\n",
     "    for data in loader:\n        data = data.to(DEVICE)\n        out = model(data.x,data.edge_index,data.edge_attr,data.batch)\n",
     "run(): move batch to selected device"),

    ("models/STAG/cross_validation.py",
     "            loss = criterion(out, data.y.float().to('cuda'))\n",
     "            loss = criterion(out, data.y.float().to(DEVICE))\n",
     "run(): loss target on selected device"),

    ("models/STAG/cross_validation.py",
     "        stag = STAG().to('cuda')\n",
     "        stag = STAG().to(DEVICE)\n",
     "cross_validation(): model on selected device"),

    ("models/STAG/cross_validation.py",
     "        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1-np.mean(df['label']))/np.mean(df['label'])))\n",
     "        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1-np.mean(df['label']))/np.mean(df['label']),dtype=torch.float32).to(DEVICE))\n",
     "cross_validation(): pos_weight on selected device (and float32 for mps)"),

    ("models/STAG/cross_validation.py",
     "                        for data in test_loader:\n                            data = data.to('cuda')\n",
     "                        for data in test_loader:\n                            data = data.to(DEVICE)\n",
     "best-epoch prediction dump: batch on selected device"),

    # pandas 2.x removed DataFrame.append
    ("models/STAG/cross_validation.py",
     "                        final_preds = pd.DataFrame(columns=['index','label','pred'])\n"
     "                        for j in range(len(pred)):\n"
     "                            final_preds = final_preds.append({'index':y_test_ids[j],'label':y_test_targets[j],'pred':pred[j]},ignore_index=True)\n",
     "                        final_preds = pd.DataFrame({'index':y_test_ids[:len(pred)],'label':y_test_targets,'pred':pred})\n",
     "replace removed DataFrame.append (pandas >= 2.0) with a direct frame"),

    ("models/STAG/cross_validation.py",
     "            torch.cuda.empty_cache()\n",
     "            _empty_cache()\n",
     "guard cache clearing for non-cuda devices"),

    # PyG >= 2.5 no longer promotes a plain python/numpy scalar `y` to a tensor,
    # so batching yields a list and data.y.float() raises. Store y as a 1-element
    # tensor, which collates to shape [batch] exactly as the loss code expects.
    ("models/STAG/cross_validation.py",
     "        train_data_list = [torch_geometric.data.Data(x=graph.get_node_data(),edge_index=graph.edge_index,edge_attr=graph.edge_attr,y=graph.label) for graph in train_graphs]\n",
     "        train_data_list = [torch_geometric.data.Data(x=graph.get_node_data(),edge_index=graph.edge_index,edge_attr=graph.edge_attr,y=torch.tensor([graph.label],dtype=torch.float32)) for graph in train_graphs]\n",
     "store train y as a tensor (PyG >= 2.5 no longer promotes scalars)"),

    ("models/STAG/cross_validation.py",
     "        test_data_list = [torch_geometric.data.Data(x=graph.get_node_data(),edge_index=graph.edge_index,edge_attr=graph.edge_attr,y=graph.label) for graph in test_graphs]\n",
     "        test_data_list = [torch_geometric.data.Data(x=graph.get_node_data(),edge_index=graph.edge_index,edge_attr=graph.edge_attr,y=torch.tensor([graph.label],dtype=torch.float32)) for graph in test_graphs]\n",
     "store test y as a tensor (PyG >= 2.5 no longer promotes scalars)"),

    # ---- encode_structure.py: torch-cluster-free radius_graph ----
    ("models/STAG/encode_structure.py",
     "from .utils import *\n",
     "from .utils import *\n"
     "\n"
     "# --- local: radius_graph without torch-cluster ---\n"
     "def _radius_graph_fallback(pos, r, loop=False):\n"
     "    \"\"\"Dense pairwise equivalent of torch_cluster.radius_graph.\n"
     "    Same output semantics (edge_index, both directions, self-loops optional).\n"
     "    Graphs here are a few hundred residues, so the dense distance matrix is cheap.\"\"\"\n"
     "    d = torch.cdist(pos, pos)\n"
     "    mask = d < r\n"
     "    if not loop:\n"
     "        mask.fill_diagonal_(False)\n"
     "    row, col = mask.nonzero(as_tuple=True)\n"
     "    return torch.stack([col, row], dim=0)\n"
     "\n"
     "def _radius_graph(pos, r, loop=False):\n"
     "    try:\n"
     "        return gnn.radius_graph(pos, r=r, loop=loop)\n"
     "    except ImportError:\n"
     "        return _radius_graph_fallback(pos, r, loop=loop)\n"
     "# --- end local ---\n",
     "add a pure-torch radius_graph fallback for when torch-cluster is absent"),

    ("models/STAG/encode_structure.py",
     "    res_graph.set_edge_index(gnn.radius_graph(res_graph.get_coords(), r=edge_radius, loop=False))\n    return res_graph",
     "    res_graph.set_edge_index(_radius_graph(res_graph.get_coords(), r=edge_radius, loop=False))\n    return res_graph",
     "use the fallback-aware radius_graph"),
]


def main():
    if not (REPO / "evaluations.py").is_file():
        sys.exit(f"not a STAG_public checkout: {REPO}")

    applied, skipped = [], []
    for rel, old, new, desc in EDITS:
        p = REPO / rel
        text = p.read_text()
        if new in text:
            skipped.append(f"{rel}: already applied - {desc}")
            continue
        if old not in text:
            sys.exit(f"ABORT {rel}: upstream text not found for '{desc}'.\n"
                     f"The repo differs from the reviewed commit; patch not applied.")
        if text.count(old) != 1:
            sys.exit(f"ABORT {rel}: '{desc}' matched {text.count(old)} times, expected 1.")
        p.write_text(text.replace(old, new))
        applied.append(f"{rel}: {desc}")

    (REPO / "output").mkdir(exist_ok=True)

    print(f"applied {len(applied)} edit(s):")
    for a in applied:
        print("  +", a)
    for s in skipped:
        print("  =", s)


if __name__ == "__main__":
    main()
