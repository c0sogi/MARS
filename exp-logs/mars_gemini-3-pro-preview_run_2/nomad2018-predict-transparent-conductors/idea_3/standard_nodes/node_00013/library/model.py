import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import softmax
from library.config import Config


class GaussianRBF(nn.Module):
    """
    Expands scalar distances into a vector of Gaussian Radial Basis Functions.
    This provides a continuous, high-resolution representation of interatomic distances.
    """

    def __init__(self, num_bins, cutoff):
        super().__init__()
        self.num_bins = num_bins
        self.cutoff = cutoff
        # Register centers as buffers so they are part of the state dict but not trainable parameters
        # Linearly spaced centers from 0 to cutoff
        centers = torch.linspace(0, cutoff, num_bins)
        self.register_buffer("centers", centers)

        # Width (sigma) determined by the spacing between centers
        if num_bins > 1:
            width = centers[1] - centers[0]
        else:
            width = 1.0
        # gamma = 1 / (2 * sigma^2), approximating sigma with width
        self.gamma = 1.0 / (width**2)

    def forward(self, distances):
        """
        Args:
            distances: Tensor of shape (num_edges,) representing edge lengths.
        Returns:
            Tensor of shape (num_edges, num_bins)
        """
        # Expand dimensions for broadcasting: (E, 1) - (1, Bins) -> (E, Bins)
        diff = distances.unsqueeze(1) - self.centers.unsqueeze(0)
        return torch.exp(-self.gamma * (diff**2))


class CGCNNConv(MessagePassing):
    """
    Crystal Graph Convolutional Layer.
    Cite solution_lesson_node_00011: Inductive Bias in Small-Data Regimes
    """

    def __init__(self, node_dim, edge_dim):
        super().__init__(aggr="add")
        self.emb_dim = node_dim
        self.edge_dim = edge_dim

        # Input to linear layers is concatenation of node_i, node_j, and edge_attr
        input_dim = 2 * node_dim + edge_dim

        self.lin_f = nn.Linear(input_dim, node_dim)
        self.lin_s = nn.Linear(input_dim, node_dim)
        self.bn = nn.BatchNorm1d(node_dim)

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.bn(out) + x

    def message(self, x_i, x_j, edge_attr):
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        # Gated convolution formula
        gate = torch.sigmoid(self.lin_f(z))
        core = F.softplus(self.lin_s(z))
        return gate * core


class CGCNN(nn.Module):
    """
    Crystal Graph Convolutional Neural Network (CGCNN).
    Replaces DBGT based on Lesson 11.
    """

    def __init__(self, config=Config):
        super().__init__()

        self.embed_dim = config.EMBEDDING_DIM
        self.rbf_bins = config.RBF_BINS
        self.cutoff = config.CUTOFF_RADIUS

        # Node Embedding
        self.node_embedding = nn.Embedding(config.MAX_ATOMIC_NUMBER + 1, self.embed_dim)

        # Edge RBF Featurizer
        self.rbf = GaussianRBF(self.rbf_bins, self.cutoff)

        # Stack of CGCNN Layers
        self.layers = nn.ModuleList(
            [
                CGCNNConv(
                    node_dim=self.embed_dim,
                    edge_dim=self.rbf_bins,
                )
                for _ in range(config.N_LAYERS)
            ]
        )

        # Output Head
        self.output_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, len(config.TARGET_COLS)),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, data):
        # Unpack data
        if isinstance(data, dict):
            x = data["x"]
            edge_index = data["edge_index"]
            edge_dists = data["edge_attr"]
            batch = data["batch"]
        else:
            x = data.x
            edge_index = data.edge_index
            edge_dists = data.edge_attr
            batch = data.batch

        # 1. Node Embedding
        h = self.node_embedding(x)

        # 2. Edge Featurization (RBF)
        edge_rbf = self.rbf(edge_dists)

        # 3. CGCNN Layers
        for layer in self.layers:
            h = layer(h, edge_index, edge_rbf)

        # 4. Global Pooling
        h_graph = global_mean_pool(h, batch)

        # 5. Output Prediction
        out = self.output_head(h_graph)

        return out
