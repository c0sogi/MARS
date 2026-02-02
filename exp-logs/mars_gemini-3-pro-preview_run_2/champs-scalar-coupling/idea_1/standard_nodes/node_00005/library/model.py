import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Batch
from library.config import Config
import math


class GaussianSmearing(nn.Module):
    """
    Expands scalar distances into a vector of Radial Basis Functions (RBF).
    Cite solution_lesson_node_00004: "expand continuous distances into high-dimensional vectors"
    """

    def __init__(
        self, start=0.0, stop=Config.CUTOFF_RADIUS, num_gaussians=Config.NUM_GAUSSIANS
    ):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / ((stop - start) / (num_gaussians - 1)) ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class InteractionBlock(MessagePassing):
    """
    Continuous Filter Convolution Block (SchNet).
    Cite solution_lesson_node_00004: "Interaction Blocks (Continuous Filter Convolutions)"
    """

    def __init__(self, hidden_channels, num_gaussians, num_filters):
        super().__init__(aggr="add")
        self.mlp = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            nn.Softplus(),
            nn.Linear(num_filters, num_filters),
        )
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr):
        # edge_attr is the RBF expansion of distance
        W = self.mlp(edge_attr)
        return self.propagate(edge_index, x=x, W=W)

    def message(self, x_j, W):
        # Continuous filter convolution: element-wise multiplication
        return x_j * W

    def update(self, aggr_out, x):
        # Residual connection
        return x + self.lin(aggr_out)


class SchNetModel(nn.Module):
    """
    SchNet-based architecture for Scalar Coupling Prediction.
    Replaces DistanceWeightedGCN to capture geometric dependencies via RBFs.
    """

    def __init__(self):
        super(SchNetModel, self).__init__()

        # Hyperparameters
        self.atom_embed_dim = Config.ATOM_EMBED_DIM
        self.hidden_dim = Config.HIDDEN_DIM
        self.num_interactions = Config.NUM_INTERACTIONS
        self.num_gaussians = Config.NUM_GAUSSIANS
        self.num_filters = Config.NUM_FILTERS
        self.type_embed_dim = Config.TYPE_EMBED_DIM
        self.mlp_hidden_dim = Config.MLP_HIDDEN_DIM

        # 1. Embeddings & Expansion
        self.atom_embedding = nn.Embedding(len(Config.ATOM_MAP), self.hidden_dim)
        self.distance_expansion = GaussianSmearing(
            0.0, Config.CUTOFF_RADIUS, self.num_gaussians
        )
        self.type_embedding = nn.Embedding(len(Config.TYPE_MAP), self.type_embed_dim)

        # 2. Interaction Blocks
        self.interactions = nn.ModuleList()
        for _ in range(self.num_interactions):
            self.interactions.append(
                InteractionBlock(self.hidden_dim, self.num_gaussians, self.num_filters)
            )

        # 3. Readout Head
        # We concatenate: Node_i, Node_j, RBF(dist_ij), Type_Emb
        input_dim = (self.hidden_dim * 2) + self.num_gaussians + self.type_embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.mlp_hidden_dim),
            nn.Softplus(),  # SchNet typically uses Softplus (smooth ReLU)
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
            nn.Softplus(),
            nn.Linear(self.mlp_hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # --- 1. Representation Learning ---
        h = self.atom_embedding(x)

        # Expand distances (edge_attr contains raw distances)
        edge_rbf = self.distance_expansion(edge_attr.squeeze(-1))

        # Apply Interaction Blocks
        for interaction in self.interactions:
            h = interaction(h, edge_index, edge_rbf)

        # --- 2. Pairwise Feature Extraction ---
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

        # Calculate Distance RBF for the specific pairs we are predicting
        pos0 = data.pos[idx0]
        pos1 = data.pos[idx1]
        dist = torch.norm(pos0 - pos1, p=2, dim=-1)
        dist_rbf = self.distance_expansion(dist)

        type_emb = self.type_embedding(data.couple_type)

        # --- 3. Prediction ---
        out = torch.cat([h0, h1, dist_rbf, type_emb], dim=-1)
        pred = self.mlp(out)

        return pred
