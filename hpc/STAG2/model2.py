"""Two encoders the STAG paper does not have, added without touching models/STAG.

Both come from Abeer, Roy, Qian & Yoon, npj Drug Discovery 3:31 (2026), which
evaluated them on the same structure dataset and reported that an E(n)-
equivariant encoder was more robust than STAG's own EVTConv, and that a dual
encoder reading the TCR-MHC subgraph alongside the full interface helped on
some peptide clusters.

STAG_EGNN  E(n)-equivariant message passing. The message carries the SQUARED
           DISTANCE between the two residues rather than their coordinates, so
           the network is invariant to rotation and translation by construction
           instead of having to learn it. Coordinates are not updated -- for a
           graph-level classifier the coordinate update of Satorras et al. buys
           nothing and costs stability.

STAG_DUAL  Two encoders over the same complex: the whole TCR-pMHC interface,
           and the subgraph with the peptide removed (chains M, A, B). A
           semantic term pulls the two graph embeddings together for binders
           and pushes them apart for non-binders, so the model is told that the
           peptide's contribution is what should differ between the classes.
           Classification reads both embeddings.

Everything else -- feature widths, dropout, LayerNorm placement, pooling and
the classifier head -- is copied from models/STAG/model.py so that a difference
in the result is a difference in the encoder and not in the scaffold.
"""
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptTensor
from torch_geometric.utils import subgraph

H = 32          # hidden width, same as STAG
E = 16          # edge width, same as STAG
NODE_IN = 19
EDGE_IN = 19
DROP = 0.125


class EGNNConv(MessagePassing):
    """E(n)-equivariant convolution, coordinates held fixed."""

    def __init__(self, hidden: int = H, edge_dim: int = E):
        super().__init__(aggr='add', node_dim=0)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden + 1 + edge_dim, hidden), nn.LayerNorm(hidden),
            nn.LeakyReLU(), nn.Dropout(DROP), nn.Linear(hidden, hidden))
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden),
            nn.LeakyReLU(), nn.Dropout(DROP), nn.Linear(hidden, hidden))

    def forward(self, x: Tensor, pos: Tensor, edge_index, edge_attr: OptTensor):
        # propagate_type: (x: Tensor, pos: Tensor, edge_attr: OptTensor)
        m = self.propagate(edge_index, x=x, pos=pos, edge_attr=edge_attr)
        return self.node_mlp(torch.cat([x, m], dim=-1))

    def message(self, x_i: Tensor, x_j: Tensor, pos_i: Tensor, pos_j: Tensor,
                edge_attr: Tensor) -> Tensor:
        d2 = ((pos_i - pos_j) ** 2).sum(dim=-1, keepdim=True)
        return self.edge_mlp(torch.cat([x_i, x_j, d2, edge_attr], dim=-1))


class _Stem(nn.Module):
    """The node/edge embedding stem of models/STAG/model.py, unchanged."""

    def __init__(self):
        super().__init__()
        self.node_lin1 = nn.Linear(NODE_IN, H)
        self.node_norm = gnn.norm.LayerNorm(H)
        self.node_lin2 = nn.Linear(H, H)
        self.edge_lin1 = nn.Linear(H * 2 + EDGE_IN, E)
        self.edge_norm = nn.LayerNorm(E)
        self.edge_lin2 = nn.Linear(E, E)

    def forward(self, x, edge_index, edge_attr, batch):
        row, col = edge_index
        x = self.node_lin1(x)
        x = self.node_norm(x, batch)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=DROP, training=self.training)
        x = self.node_lin2(x)
        e = torch.cat([x[row], x[col], edge_attr], dim=-1)
        e = self.edge_lin1(e)
        e = self.edge_norm(e)
        e = F.leaky_relu(e)
        e = F.dropout(e, p=DROP, training=self.training)
        e = self.edge_lin2(e)
        return x, e


class _EGNNEncoder(nn.Module):
    """Stem + three EGNN blocks + mean pooling -> one graph embedding."""

    def __init__(self):
        super().__init__()
        self.stem = _Stem()
        self.conv = nn.ModuleList([EGNNConv() for _ in range(3)])
        self.norm = nn.ModuleList([gnn.norm.LayerNorm(H) for _ in range(3)])

    def forward(self, x, pos, edge_index, edge_attr, batch):
        x, e = self.stem(x, edge_index, edge_attr, batch)
        for i, (conv, norm) in enumerate(zip(self.conv, self.norm)):
            x = conv(x, pos, edge_index, e)
            x = norm(x, batch)
            if i < 2:
                x = F.leaky_relu(x)
        return gnn.global_mean_pool(x, batch)


class _Head(nn.Module):
    """The classifier head of models/STAG/model.py, widened only by its input."""

    def __init__(self, in_dim: int = H):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, H)
        self.norm = nn.LayerNorm(H)
        self.lin2 = nn.Linear(H, 1)

    def forward(self, g):
        g = self.lin1(g)
        g = self.norm(g)
        g = F.leaky_relu(g)
        g = F.dropout(g, p=DROP, training=self.training)
        return self.lin2(g)


class STAG_EGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = _EGNNEncoder()
        self.head = _Head(H)

    def forward(self, data):
        g = self.enc(data.x, data.pos, data.edge_index, data.edge_attr, data.batch)
        return self.head(g).squeeze(), None


class STAG_DUAL(nn.Module):
    """Full interface graph + the same graph with the peptide removed.

    Returns (logit, semantic) so the training loop can add the semantic term;
    `semantic` is the cosine similarity of the two graph embeddings, which the
    loop turns into a loss using the labels.
    """

    def __init__(self):
        super().__init__()
        self.enc_full = _EGNNEncoder()
        self.enc_sub = _EGNNEncoder()
        self.head = _Head(H * 2)

    def forward(self, data):
        g_full = self.enc_full(data.x, data.pos, data.edge_index,
                               data.edge_attr, data.batch)
        keep = ~data.is_pep
        sub_ei, sub_ea = subgraph(keep, data.edge_index, data.edge_attr,
                                  relabel_nodes=True, num_nodes=data.x.size(0))
        g_sub = self.enc_sub(data.x[keep], data.pos[keep], sub_ei, sub_ea,
                             data.batch[keep])
        if g_sub.size(0) != g_full.size(0):        # a graph with no TCR/MHC node
            pad = g_full.new_zeros(g_full.size(0) - g_sub.size(0), g_sub.size(1))
            g_sub = torch.cat([g_sub, pad], dim=0)
        sem = F.cosine_similarity(g_full, g_sub, dim=-1)
        return self.head(torch.cat([g_full, g_sub], dim=-1)).squeeze(), sem


def semantic_loss(sem, y, margin: float = 0.0):
    """Binders: the two views should agree. Non-binders: they should not.

    The peptide is what the subgraph throws away, so for a real binder the
    peptide-free view should still carry the same signal, and for a non-binder
    it should not. Hinged at `margin` so the push-apart term stops once the
    two embeddings are orthogonal rather than driving them to anti-correlation.
    """
    y = y.float()
    return (y * (1.0 - sem) + (1.0 - y) * torch.clamp(sem - margin, min=0.0)).mean()
