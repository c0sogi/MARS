import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv, global_mean_pool, global_max_pool
from library.config import Config


class GaussianRBF(nn.Module):
    """
    Gaussian Radial Basis Function expansion for edge distances.
    Expands scalar distances into a vector of RBF evaluations.
    """

    def __init__(self, start=0.0, stop=5.0, n_centers=60):
        super().__init__()
        self.centers = torch.linspace(start, stop, n_centers)
        self.width = (stop - start) / n_centers
        self.gamma = 1.0 / (self.width**2)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [num_edges] containing distances.
        Returns:
            Tensor of shape [num_edges, n_centers]
        """
        # Ensure centers are on the same device as input
        centers = self.centers.to(x.device)
        return torch.exp(-self.gamma * (x.unsqueeze(-1) - centers) ** 2)


class ScalarResidualCGConv(nn.Module):
    """
    Custom Crystal Graph Convolution layer with Scalar Learnable Residuals.

    Update rule:
    h_{l+1} = Softplus(BatchNorm(CGConv(h_l, e_{ij}) + (1 + epsilon) * h_l))

    Where epsilon is a learnable scalar parameter initialized to 0.0.
    """

    def __init__(self, channels, edge_dim, dropout_rate=0.1):
        super().__init__()
        # CGConv performs the gated message passing
        # We disable internal batch_norm to apply it explicitly after residual addition
        self.conv = CGConv(channels, dim=edge_dim, batch_norm=False, bias=True)

        self.bn = nn.BatchNorm1d(channels)
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(dropout_rate)

        # Learnable scalar for residual connection
        self.epsilon = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, edge_index, edge_attr):
        # 1. Convolution (Message Passing)
        conv_out = self.conv(x, edge_index, edge_attr)

        # 2. Scalar Learnable Residual
        # (1 + epsilon) * Identity
        res_out = (1.0 + self.epsilon) * x

        # 3. Combine
        out = conv_out + res_out

        # 4. BatchNorm
        out = self.bn(out)

        # 5. Activation
        out = self.softplus(out)

        # 6. Dropout
        out = self.dropout(out)

        return out


class SR_CGN_DP(nn.Module):
    """
    Scalar-Residual Crystal Graph Network with Dual-Pooling (SR-CGN-DP).

    Architecture:
    1. Node Embedding (Atomic Number -> Vector)
    2. Edge Expansion (Distance -> RBF -> Vector)
    3. Stack of ScalarResidualCGConv layers
    4. Dual Global Pooling (Mean + Max)
    5. Decoupled MLP Heads for Formation Energy and Bandgap
    """

    def __init__(
        self,
        node_dim=Config.ATOM_EMBEDDING_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout_rate=Config.DROPOUT_RATE,
        rbf_bins=Config.RBF_BINS,
        rbf_min=Config.RBF_MIN,
        rbf_max=Config.RBF_MAX,
    ):
        super().__init__()

        # 1. Embeddings
        # Atomic numbers go up to ~100. We allocate a safe margin.
        self.node_embedding = nn.Embedding(100, node_dim)

        self.rbf = GaussianRBF(start=rbf_min, stop=rbf_max, n_centers=rbf_bins)
        self.edge_projection = nn.Linear(rbf_bins, node_dim)

        # 2. Backbone
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                ScalarResidualCGConv(
                    channels=node_dim, edge_dim=node_dim, dropout_rate=dropout_rate
                )
            )

        # 3. Readout (Dual Pooling)
        # Concatenates Mean and Max pooling, doubling the feature dimension
        self.pool_dim = node_dim * 2

        # 4. Prediction Heads
        # Head 1: Formation Energy
        self.fc_formation = nn.Sequential(
            nn.Linear(self.pool_dim, 128),
            nn.Softplus(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.Softplus(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
        )

        # Head 2: Bandgap Energy
        self.fc_bandgap = nn.Sequential(
            nn.Linear(self.pool_dim, 128),
            nn.Softplus(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.Softplus(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        """
        Args:
            data: PyG Data object containing:
                - x: Atomic numbers [num_nodes]
                - edge_index: Graph connectivity [2, num_edges]
                - edge_attr: Edge distances [num_edges, 1]
                - batch: Batch index [num_nodes]
        Returns:
            Tensor of shape [batch_size, 2] (formation_energy, bandgap_energy)
        """
        # Node Embedding
        h = self.node_embedding(data.x)  # [num_nodes, node_dim]

        # Edge Feature Processing
        # Squeeze edge_attr to [num_edges] for RBF
        edge_dist = data.edge_attr.squeeze(-1)
        edge_feat = self.rbf(edge_dist)  # [num_edges, rbf_bins]
        edge_feat = self.edge_projection(edge_feat)  # [num_edges, node_dim]

        # Message Passing Layers
        for layer in self.layers:
            h = layer(h, data.edge_index, edge_feat)

        # Dual Global Pooling
        # Aggregate node features to graph-level features
        h_mean = global_mean_pool(h, data.batch)  # [batch_size, node_dim]
        h_max = global_max_pool(h, data.batch)  # [batch_size, node_dim]

        h_global = torch.cat([h_mean, h_max], dim=1)  # [batch_size, node_dim * 2]

        # Prediction Heads
        out_formation = self.fc_formation(h_global)  # [batch_size, 1]
        out_bandgap = self.fc_bandgap(h_global)  # [batch_size, 1]

        # Concatenate outputs
        return torch.cat([out_formation, out_bandgap], dim=1)
