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


class CompositionalMLP(nn.Module):
    """
    Predicts property baselines purely from atomic composition.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class CGCNNConv(MessagePassing):
    """
    Crystal Graph Convolutional Neural Network Layer.
    Updates node embeddings using gated edge messages.
    """

    def __init__(self, node_dim, edge_dim):
        super().__init__(aggr="add")
        self.node_dim = node_dim
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


class StructuralCGCNN(nn.Module):
    """
    Graph Neural Network to predict property residuals from crystal structure.
    """

    def __init__(self, node_dim, edge_dim, n_layers, output_dim, dropout):
        super().__init__()
        # Embedding for atomic numbers (1-100)
        self.embedding = nn.Embedding(100, node_dim)

        # Edge expansion
        self.edge_expansion = GaussianSmearing(0.0, CUTOFF_RADIUS, edge_dim)

        # Stack of Graph Convolutions
        self.convs = nn.ModuleList(
            [CGCNNConv(node_dim, edge_dim) for _ in range(n_layers)]
        )

        self.dropout = nn.Dropout(dropout)

        # Readout function to map graph embedding to residual
        self.readout = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, output_dim),
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

        # Global Pooling
        h_graph = global_mean_pool(h, batch)

        # Predict residuals
        out = self.readout(h_graph)
        return out


class ACSRNet(nn.Module):
    """
    Additive Composition-Structure Residual Network.

    Prediction = Baseline(Composition) + Residual(Structure)
    """

    def __init__(self):
        super().__init__()

        # 1. Compositional Baseline Stream
        self.comp_mlp = CompositionalMLP(
            input_dim=len(COMPOSITION_COLS),
            hidden_dim=EMBEDDING_DIM,
            output_dim=len(TARGET_COLS),
            dropout=DROPOUT,
        )

        # 2. Structural Residual Stream
        self.struct_gnn = StructuralCGCNN(
            node_dim=EMBEDDING_DIM,
            edge_dim=N_RBF,
            n_layers=N_CONV_LAYERS,
            output_dim=len(TARGET_COLS),
            dropout=DROPOUT,
        )

    def forward(self, composition, data):
        """
        Args:
            composition (Tensor): [Batch, n_comp_features] - Standardized atomic fractions.
            data (Data): PyG DataBatch object containing graph structure.

        Returns:
            Tensor: [Batch, n_targets] - Predicted log-standardized properties.
        """
        # Predict baseline from composition
        baseline = self.comp_mlp(composition)

        # Predict residual from structure
        residual = self.struct_gnn(data)

        # Additive fusion
        prediction = baseline + residual

        return prediction
