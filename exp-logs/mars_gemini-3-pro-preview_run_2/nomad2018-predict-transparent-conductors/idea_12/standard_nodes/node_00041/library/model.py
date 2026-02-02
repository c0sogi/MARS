import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from library.config import (
    EMBEDDING_DIM,
    N_CONV_LAYERS,
    N_RBF,
    DROPOUT,
    CUTOFF_RADIUS,
    COMPOSITION_COLS,
    TARGET_COLS,
)


class GaussianSmearing(nn.Module):
    """
    Expands distances using a set of Gaussian radial basis functions.
    """

    def __init__(self, start=0.0, stop=CUTOFF_RADIUS, n_gaussians=N_RBF):
        super().__init__()
        offset = torch.linspace(start, stop, n_gaussians)
        # The width is determined by the spacing between centers
        self.coeff = -0.5 / ((stop - start) / (n_gaussians - 1)) ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        # dist: [num_edges] -> [num_edges, n_gaussians]
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class CGCNNConv(MessagePassing):
    """
    Crystal Graph Convolutional Neural Network Layer.
    Updates node embeddings using gated edge messages.
    """

    def __init__(self, node_dim, edge_dim):
        super().__init__(aggr="add")
        self.emb_dim = node_dim
        self.edge_dim = edge_dim

        # Input to linear layers involves node_i, node_j, and edge_ij
        input_dim = 2 * node_dim + edge_dim

        self.linear_f = nn.Linear(input_dim, node_dim)
        self.linear_s = nn.Linear(input_dim, node_dim)
        self.bn = nn.BatchNorm1d(node_dim)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # Concatenate source node, target node, and edge features
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)

        # Gated convolution: sigmoid(W_f * z) * softplus(W_s * z)
        gate = torch.sigmoid(self.linear_f(z))
        filter_ = F.softplus(self.linear_s(z))

        return gate * filter_

    def update(self, aggr_out, x):
        # Residual connection and batch normalization
        return self.bn(x + aggr_out)


class CrystalGraphConvNet(nn.Module):
    """
    Standard Crystal Graph Convolutional Neural Network (CGCNN) with decoupled heads.
    Cite {solution_lesson_node_00040}: Implicit Compositional Encoding in Wide GNNs Supersedes Explicit Residual Architectures.
    Cite {solution_lesson_node_00039}: Capacity and Task Decoupling in Multi-Target Crystal Property Prediction.
    """

    def __init__(self):
        super().__init__()
        # Hyperparameters from config
        node_dim = EMBEDDING_DIM
        edge_dim = N_RBF
        n_layers = N_CONV_LAYERS
        dropout = DROPOUT

        # Embedding for atomic numbers (1-100)
        self.embedding = nn.Embedding(100, node_dim)

        # Edge expansion
        self.edge_expansion = GaussianSmearing(0.0, CUTOFF_RADIUS, edge_dim)

        # Stack of Graph Convolutions
        self.convs = nn.ModuleList(
            [CGCNNConv(node_dim, edge_dim) for _ in range(n_layers)]
        )

        self.dropout = nn.Dropout(dropout)

        # Decoupled Readout Heads
        # Head 1: Formation Energy
        self.head_formation = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, 1),
        )

        # Head 2: Bandgap Energy
        self.head_bandgap = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Initial node embeddings
        h = self.embedding(x)

        # Expand scalar distances to RBF vectors
        edge_features = self.edge_expansion(edge_attr.squeeze(-1))

        # Message Passing
        for conv in self.convs:
            h = conv(h, edge_index, edge_features)
            h = self.dropout(h)

        # Global Pooling (Mean) - Cite {solution_lesson_node_00033}
        h_graph = global_mean_pool(h, batch)

        # Predict targets using separate heads
        out_formation = self.head_formation(h_graph)
        out_bandgap = self.head_bandgap(h_graph)

        # Concatenate outputs [Batch, 2]
        return torch.cat([out_formation, out_bandgap], dim=1)
