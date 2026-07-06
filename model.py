"""
MolScreen GNN: a message-passing graph neural network for multi-label
molecular toxicity prediction on Tox21.

Architecture: stacked GINEConv layers (Graph Isomorphism Network with
Edge features) -> global mean+max pooling -> MLP head -> 12 sigmoid outputs
(one per Tox21 task).

GINEConv is used (rather than plain GCN) because it incorporates edge
features (bond type, conjugation, ring membership) directly into message
passing, which matters for toxicity prediction since functional group
connectivity (not just atom identity) drives toxicophore behavior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool
from featurize import node_feature_dim, edge_feature_dim


class MolScreenGNN(nn.Module):
    def __init__(self, node_dim=None, edge_dim=None, hidden_dim=128,
                 num_layers=4, num_tasks=12, dropout=0.2):
        super().__init__()

        node_dim = node_dim or node_feature_dim()
        edge_dim = edge_dim or edge_feature_dim()

        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = dropout

        # pooled representation is mean+max concat -> 2*hidden_dim
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tasks),
        )

    def forward(self, x, edge_index, edge_attr, batch, return_node_repr=False):
        x = self.node_embed(x)
        edge_attr = self.edge_embed(edge_attr)

        for conv, bn in zip(self.convs, self.bns):
            x_new = conv(x, edge_index, edge_attr)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x + x_new  # residual connection for stability across layers

        node_repr = x  # per-atom representation, used later for explainability

        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        pooled = torch.cat([mean_pool, max_pool], dim=1)

        out = self.head(pooled)  # raw logits, shape (batch, num_tasks)

        if return_node_repr:
            return out, node_repr
        return out


def masked_bce_loss(logits, targets):
    """
    Multi-label BCE loss that ignores missing labels (NaN in Tox21).
    Tox21 has substantial missingness per task, so this masking is
    required, not optional -- treating missing as negative would bias
    the model toward predicting "non-toxic" on tasks with high missingness.
    """
    mask = ~torch.isnan(targets)
    targets_filled = torch.where(mask, targets, torch.zeros_like(targets))
    loss = F.binary_cross_entropy_with_logits(logits, targets_filled, reduction='none')
    loss = loss * mask.float()
    # average only over valid (non-missing) entries
    return loss.sum() / mask.float().sum().clamp(min=1.0)


if __name__ == "__main__":
    # smoke test with a tiny batch
    from torch_geometric.loader import DataLoader
    import pickle

    with open('tox21_graphs.pkl', 'rb') as f:
        graphs = pickle.load(f)

    loader = DataLoader(graphs[:8], batch_size=8)
    batch = next(iter(loader))

    model = MolScreenGNN()
    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    print("Output shape:", out.shape)  # should be (8, 12)
    print("Sample logits:", out[0])

    loss = masked_bce_loss(out, batch.y)
    print("Loss:", loss.item())

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params:,}")
