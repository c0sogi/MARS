import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Batch
from library.config import Config


import math
from torch_scatter import scatter


class GaussianSmearing(nn.Module):
    """
    Expands a scalar distance into a vector of radial basis functions (RBF).
    Cite solution_lesson_node_00002: Replaces scalar distance features with high-resolution RBFs.
    """

    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / ((stop - start) / (num_gaussians - 1)) ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class InteractionBlock(nn.Module):
    """
    Continuous Filter Convolution block similar to SchNet.
    Aggregates information from neighbors using distance-based filters.
    """

    def __init__(self, hidden_channels, num_gaussians, num_filters):
        super(InteractionBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            nn.ReLU(),
            nn.Linear(num_filters, num_filters),
        )
        self.conv = nn.Linear(hidden_channels, num_filters, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.conv.weight)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

    def forward(self, x, edge_index, edge_attr):
        # edge_attr is the RBF expanded distance

        # Filter generation: MLP(RBF)
        W = self.mlp(edge_attr)

        # Continuous filter convolution
        # x_j * W_ij
        row, col = edge_index
        x_j = x[col]
        x_j_trans = self.conv(x_j)

        # Element-wise product and aggregation
        m = x_j_trans * W
        out = scatter(m, row, dim=0, dim_size=x.size(0), reduce="add")

        return out


class SchNetModel(nn.Module):
    """
    SchNet-like architecture for predicting scalar coupling constants.
    Uses Gaussian Smearing and Interaction Blocks to capture geometric dependencies.
    """

    def __init__(self):
        super(SchNetModel, self).__init__()

        # Config
        self.hidden_dim = Config.HIDDEN_DIM
        self.num_filters = Config.NUM_FILTERS
        self.num_interactions = Config.NUM_INTERACTIONS
        self.num_gaussians = Config.NUM_GAUSSIANS

        # Embeddings
        self.atom_embedding = nn.Embedding(len(Config.ATOM_MAP), self.hidden_dim)
        self.type_embedding = nn.Embedding(len(Config.TYPE_MAP), Config.TYPE_EMBED_DIM)

        # Distance Expansion
        self.distance_expansion = GaussianSmearing(
            Config.RBF_MIN, Config.RBF_MAX, self.num_gaussians
        )

        # Interaction Blocks
        self.interactions = nn.ModuleList(
            [
                InteractionBlock(self.hidden_dim, self.num_gaussians, self.num_filters)
                for _ in range(self.num_interactions)
            ]
        )

        # Output layers for interaction blocks (residual update)
        self.lin_updates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.num_filters, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
                for _ in range(self.num_interactions)
            ]
        )

        # Readout MLP
        # Input: Node_i + Node_j + RBF(dist) + Type
        input_dim = (self.hidden_dim * 2) + self.num_gaussians + Config.TYPE_EMBED_DIM

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, Config.MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.MLP_HIDDEN_DIM, Config.MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.MLP_HIDDEN_DIM, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # 1. Initial Embedding
        h = self.atom_embedding(x)

        # 2. Distance Expansion
        # edge_attr from data is raw distance [Num_Edges, 1]
        edge_rbf = self.distance_expansion(edge_attr.squeeze(-1))

        # 3. Interaction Blocks
        for interaction, update in zip(self.interactions, self.lin_updates):
            v = interaction(h, edge_index, edge_rbf)
            h = h + update(v)

        # 4. Pairwise Prediction
        if isinstance(data, Batch):
            node_offsets = data.ptr[:-1]
            couple_counts = data.num_couples
            shifts = torch.repeat_interleave(node_offsets, couple_counts)
            idx0 = data.couple_idx[:, 0] + shifts
            idx1 = data.couple_idx[:, 1] + shifts
        else:
            idx0 = data.couple_idx[:, 0]
            idx1 = data.couple_idx[:, 1]

        h0 = h[idx0]
        h1 = h[idx1]

        # Re-calculate distance RBF for the specific coupling pairs
        pos0 = data.pos[idx0]
        pos1 = data.pos[idx1]
        dist = torch.norm(pos0 - pos1, p=2, dim=-1)
        dist_rbf = self.distance_expansion(dist)

        type_emb = self.type_embedding(data.couple_type)

        # Concatenate
        out = torch.cat([h0, h1, dist_rbf, type_emb], dim=-1)

        return self.mlp(out)
