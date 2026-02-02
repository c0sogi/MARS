import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv, global_mean_pool
from torch_geometric.data import Data
from library.config import Config


class GaussianSmearing(nn.Module):
    """
    Expands distances using a set of Gaussian Radial Basis Functions.
    """

    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        # The width (gamma) is determined by the spacing between centers
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        # dist: (E, 1) or (E,)
        # Returns: (E, num_gaussians)
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class InteractionBlock(nn.Module):
    """
    Interaction Block performing Gated Graph Convolution with a learnable
    scalar residual connection, Batch Normalization, and Softplus activation.
    """

    def __init__(self, hidden_dim, dropout_rate):
        super().__init__()
        # CGConv from torch_geometric implements the crystal graph convolution
        # We disable internal batch_norm to apply it explicitly as per architecture design
        self.conv = CGConv(channels=hidden_dim, dim=hidden_dim, batch_norm=False)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout_rate = dropout_rate

        # Learnable scalar epsilon for the residual connection, initialized to 0
        self.epsilon = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, edge_index, edge_attr):
        # x: (N, hidden_dim)
        # edge_index: (2, E)
        # edge_attr: (E, hidden_dim)

        # PyG's CGConv returns: x + sum(messages)
        z = self.conv(x, edge_index, edge_attr)

        # We want the update: messages + (1 + epsilon) * x
        # Since z = messages + x, we add epsilon * x to z
        out = z + self.epsilon * x

        # Apply BatchNorm, Softplus, and Dropout
        out = self.bn(out)
        out = F.softplus(out)
        out = F.dropout(out, p=self.dropout_rate, training=self.training)

        return out


class OptimizedCGCNN(nn.Module):
    """
    Optimized Crystal Graph Convolutional Network.
    Removed global skip connection to rely on deep feature refinement.
    Uses learnable residual scaling in interaction blocks.
    """

    def __init__(self, config=Config):
        super().__init__()

        self.atom_embedding_dim = config.ATOM_EMBEDDING_DIM
        self.num_rbf = config.NUM_RBF
        self.num_layers = config.NUM_GNN_LAYERS
        self.dropout_rate = config.DROPOUT_RATE
        self.cutoff = config.GRAPH_CUTOFF

        # 1. Node & Edge Embedding
        # Embedding for atomic numbers (1 to ~100)
        self.embedding = nn.Embedding(100, self.atom_embedding_dim)

        # RBF expansion for edge distances
        self.rbf = GaussianSmearing(
            start=0.0, stop=self.cutoff, num_gaussians=self.num_rbf
        )
        # Linear projection of edge features to match node dimension
        self.edge_fc = nn.Linear(self.num_rbf, self.atom_embedding_dim)

        # 2. Structural Pathway (GNN Backbone)
        self.blocks = nn.ModuleList(
            [
                InteractionBlock(self.atom_embedding_dim, self.dropout_rate)
                for _ in range(self.num_layers)
            ]
        )

        # 3. Prediction Heads
        # Input dimension is atom_embedding_dim (no concatenation)

        # Head for Formation Energy
        self.head1 = nn.Sequential(
            nn.Linear(self.atom_embedding_dim, self.atom_embedding_dim),
            nn.Softplus(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.atom_embedding_dim, 1),
        )

        # Head for Bandgap Energy
        self.head2 = nn.Sequential(
            nn.Linear(self.atom_embedding_dim, self.atom_embedding_dim),
            nn.Softplus(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.atom_embedding_dim, 1),
        )

    def forward(self, data):
        # Unpack data
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # --- Node & Edge Embedding ---
        h = self.embedding(x)  # Shape: (N, hidden_dim)

        # Edge embeddings
        edge_feat = self.rbf(edge_attr)  # Shape: (E, num_rbf)
        edge_emb = self.edge_fc(edge_feat)  # Shape: (E, hidden_dim)

        # --- Structural Pathway ---
        for block in self.blocks:
            h = block(h, edge_index, edge_emb)

        # --- Readout ---
        # Aggregate structural features
        z = global_mean_pool(h, batch)  # Shape: (Batch, hidden_dim)

        # --- Decoupled Prediction ---
        out1 = self.head1(z)  # Formation Energy
        out2 = self.head2(z)  # Bandgap Energy

        return torch.cat([out1, out2], dim=1)  # Shape: (Batch, 2)


# Alias for backward compatibility (Cite debug_lesson_10)
SS_CGCNN = OptimizedCGCNN
