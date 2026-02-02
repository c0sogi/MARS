import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from library.config import Config


class CGCNNConv(MessagePassing):
    """
    Crystal Graph Convolutional Neural Network (CGCNN) Layer.
    Implements the gated convolution:
    h_i' = h_i + sum_j [ sigmoid(W_f * z_ij + b_f) * softplus(W_s * z_ij + b_s) ]
    where z_ij = concat(h_i, h_j, e_ij).
    """

    def __init__(self, node_dim, edge_dim):
        super(CGCNNConv, self).__init__(aggr="add")
        # Fix: Rename self.node_dim to self.emb_dim to avoid conflict with MessagePassing.node_dim
        # Cite debug_lesson_7: Avoid Overwriting Reserved Attributes in Framework Subclasses
        self.emb_dim = node_dim
        self.edge_dim = edge_dim

        # Input to linears is concatenation of source node, target node, and edge features
        self.concat_dim = 2 * node_dim + edge_dim

        self.linear_f = nn.Linear(self.concat_dim, node_dim)
        self.linear_s = nn.Linear(self.concat_dim, node_dim)

        self.bn_f = nn.BatchNorm1d(node_dim)
        self.bn_s = nn.BatchNorm1d(node_dim)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # Concatenate node features and edge attributes
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)

        # Calculate filter and core parts
        f = self.linear_f(z)
        s = self.linear_s(z)

        # Apply batch norm before activation
        f = self.bn_f(f)
        s = self.bn_s(s)

        # Gated activation
        filter_out = torch.sigmoid(f)
        core_out = F.softplus(s)

        return filter_out * core_out

    def update(self, aggr_out, x):
        # Residual connection
        return x + aggr_out


class LocalStream(nn.Module):
    """
    Processes the local geometric graph structure using CGCNN layers.
    """

    def __init__(self):
        super(LocalStream, self).__init__()
        # Embedding for atomic numbers. Max Z is roughly 100.
        self.embedding = nn.Embedding(100, Config.ATOM_EMBEDDING_DIM)

        self.convs = nn.ModuleList()
        for _ in range(Config.N_LAYERS):
            self.convs.append(
                CGCNNConv(node_dim=Config.HIDDEN_DIM, edge_dim=Config.N_RBF)
            )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Initial node embedding
        h = self.embedding(x)

        # Message passing layers
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)

        # Global aggregation (Mean Pooling)
        # Mean pooling enforces the intensive property prior (energy per atom)
        h_graph = global_mean_pool(h, batch)

        return h_graph


class GlobalStream(nn.Module):
    """
    Processes global features (lattice parameters and composition) using an MLP.
    """

    def __init__(self):
        super(GlobalStream, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(Config.GLOBAL_FEATURES_DIM, Config.HIDDEN_DIM),
            nn.BatchNorm1d(Config.HIDDEN_DIM),
            nn.Softplus(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
            nn.BatchNorm1d(Config.HIDDEN_DIM),
            nn.Softplus(),
        )

    def forward(self, global_x):
        # global_x comes in as (batch_size, 1, global_dim), need to squeeze
        if global_x.dim() == 3:
            global_x = global_x.squeeze(1)
        return self.mlp(global_x)


class DualStreamNet(nn.Module):
    """
    Main architecture fusing Local and Global streams.
    """

    def __init__(self):
        super(DualStreamNet, self).__init__()
        self.local_stream = LocalStream()
        self.global_stream = GlobalStream()

        # Fusion dimension is sum of hidden dims from both streams
        fusion_dim = Config.HIDDEN_DIM * 2

        self.readout = nn.Sequential(
            nn.Linear(fusion_dim, Config.HIDDEN_DIM),
            nn.Softplus(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(
                Config.HIDDEN_DIM, 2
            ),  # Predicting 2 targets: formation_energy, bandgap
        )

    def forward(self, data):
        # Process local graph features
        h_local = self.local_stream(data)

        # Process global lattice/composition features
        h_global = self.global_stream(data.global_x)

        # Late Fusion
        h_fused = torch.cat([h_local, h_global], dim=1)

        # Final prediction
        out = self.readout(h_fused)

        return out
