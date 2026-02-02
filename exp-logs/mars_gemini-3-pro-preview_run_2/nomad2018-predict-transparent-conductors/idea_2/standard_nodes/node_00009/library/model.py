import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import softmax
from library.config import Config
from library.utils import GaussianRBF


class CGCNNConv(MessagePassing):
    """
    Crystal Graph Convolutional Neural Network Layer.
    Uses static edge features to modulate message passing.
    """

    def __init__(self, hidden_dim):
        super().__init__(aggr="add")
        # Input to linears: node_i + node_j + edge_attr = 3 * hidden_dim
        self.lin_f = nn.Linear(3 * hidden_dim, hidden_dim)
        self.lin_s = nn.Linear(3 * hidden_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # Concatenate features: [E, 3 * hidden_dim]
        z = torch.cat([x_i, x_j, edge_attr], dim=1)
        # Gated linear unit: sigmoid(W_f * z) * softplus(W_s * z)
        return torch.sigmoid(self.lin_f(z)) * F.softplus(self.lin_s(z))

    def update(self, aggr_out, x):
        # Residual connection + BatchNorm
        return self.bn(x + aggr_out)


class CrystalGraphConvNet(nn.Module):
    """
    Crystal Graph Convolutional Neural Network (CGCNN).
    """

    def __init__(self):
        super().__init__()

        self.hidden_dim = Config.HIDDEN_DIM
        self.num_layers = Config.NUM_LAYERS
        self.dropout = Config.DROPOUT

        # 1. Node Embedding
        self.node_emb = nn.Embedding(100, self.hidden_dim)

        # 2. Edge Embedding
        self.rbf = GaussianRBF(
            start=0.0, stop=Config.CUTOFF_RADIUS, n_centers=Config.NUM_RBF
        )
        self.edge_emb = nn.Linear(Config.NUM_RBF, self.hidden_dim)

        # 3. Interaction Layers
        self.layers = nn.ModuleList(
            [CGCNNConv(self.hidden_dim) for _ in range(self.num_layers)]
        )

        # 4. Output Head
        self.out_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, len(Config.TARGET_COLS)),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Initial Embeddings
        x = self.node_emb(x)
        edge_attr = self.rbf(edge_attr.squeeze(-1))
        edge_attr = self.edge_emb(edge_attr)

        # Message Passing
        for layer in self.layers:
            # CGCNNConv handles residual and norm internally in update()
            # But standard implementation often does it explicitly.
            # Here update() does x + aggr, so we just pass x.
            # However, PyG MessagePassing update() receives x as the target node features
            # if we pass it to propagate.
            # Our CGCNNConv.update takes (aggr_out, x).
            x = layer(x, edge_index, edge_attr)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global Pooling
        x_graph = global_mean_pool(x, batch)

        # Prediction
        out = self.out_mlp(x_graph)

        return out
