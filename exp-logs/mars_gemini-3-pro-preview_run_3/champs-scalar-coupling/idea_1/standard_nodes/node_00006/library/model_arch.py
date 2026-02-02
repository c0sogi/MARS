import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from library.config import (
    NUM_ATOM_TYPES,
    NUM_COUPLING_TYPES,
    EMBED_DIM,
    HIDDEN_DIM,
    NUM_GCN_LAYERS,
    NUM_MLP_LAYERS,
    DROPOUT,
    NUM_RBF,
    RADIUS_CUTOFF,
)


class GaussianSmearing(nn.Module):
    """
    Expands a distance scalar into a vector of RBF values.
    """

    def __init__(self, start=0.0, stop=RADIUS_CUTOFF, num_gaussians=NUM_RBF):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / ((stop - start) / (num_gaussians - 1)) ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class CFConvLayer(MessagePassing):
    """
    Continuous Filter Convolution Layer (SchNet-like).
    Uses RBF-expanded distances to generate edge-specific filters.
    """

    def __init__(self, in_channels, out_channels, num_filters):
        super(CFConvLayer, self).__init__(aggr="add")
        self.lin1 = nn.Linear(num_filters, num_filters)
        self.lin2 = nn.Linear(num_filters, num_filters)
        self.nn = nn.Sequential(self.lin1, nn.Softplus(), self.lin2)
        self.lin_root = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # edge_attr is the RBF expanded distance
        W = self.nn(edge_attr)
        return self.propagate(edge_index, x=x, W=W) + self.lin_root(x)

    def message(self, x_j, W):
        return x_j * W


class CouplingPredictor(nn.Module):
    """
    Improved architecture using Continuous Filter Convolutions and RBF expansion.
    """

    def __init__(self):
        super(CouplingPredictor, self).__init__()

        # 1. Embeddings & RBF
        self.atom_embedding = nn.Embedding(NUM_ATOM_TYPES, HIDDEN_DIM)
        self.distance_expansion = GaussianSmearing(0.0, RADIUS_CUTOFF, NUM_RBF)
        self.type_embedding = nn.Embedding(NUM_COUPLING_TYPES, HIDDEN_DIM)

        # 2. Interaction Layers
        self.interactions = nn.ModuleList()
        for _ in range(NUM_GCN_LAYERS):
            self.interactions.append(CFConvLayer(HIDDEN_DIM, HIDDEN_DIM, NUM_RBF))

        # 3. Prediction Head (MLP)
        # Input: Node0(D) + Node1(D) + Type(D) + Dist(1) + DotProd(1)
        mlp_input_dim = (HIDDEN_DIM * 3) + 1 + 1

        self.mlp = nn.Sequential()
        current_dim = mlp_input_dim

        for i in range(NUM_MLP_LAYERS - 1):
            self.mlp.add_module(f"lin_{i}", nn.Linear(current_dim, HIDDEN_DIM))
            self.mlp.add_module(f"bn_{i}", nn.BatchNorm1d(HIDDEN_DIM))
            self.mlp.add_module(
                f"act_{i}", nn.SiLU()
            )  # SiLU (Swish) often works better
            self.mlp.add_module(f"drop_{i}", nn.Dropout(DROPOUT))
            current_dim = HIDDEN_DIM

        self.mlp.add_module("lin_out", nn.Linear(current_dim, 1))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        target_pair = data.target_pair
        type_idx = data.type_idx
        dist = data.dist

        # --- 1. Graph Encoding ---
        # Expand distances
        edge_rbf = self.distance_expansion(edge_attr)

        # Embed atoms
        h = self.atom_embedding(x)

        # Apply Interaction layers
        for layer in self.interactions:
            h = F.silu(layer(h, edge_index, edge_rbf))

        # --- 2. Pair-Centric Readout ---
        if hasattr(data, "ptr"):
            batch_offsets = data.ptr[:-1].to(target_pair.device)
        else:
            unique, counts = torch.unique(data.batch, sorted=True, return_counts=True)
            batch_offsets = torch.cat(
                [
                    torch.tensor([0], device=target_pair.device),
                    torch.cumsum(counts, dim=0)[:-1],
                ]
            )

        global_pair_indices = target_pair + batch_offsets.view(-1, 1)

        h0 = h[global_pair_indices[:, 0]]
        h1 = h[global_pair_indices[:, 1]]

        # --- 3. Feature Fusion ---
        t_emb = self.type_embedding(type_idx)
        if t_emb.dim() == 3:
            t_emb = t_emb.squeeze(1)

        if dist.dim() == 1:
            dist = dist.unsqueeze(1)

        dot_prod = (h0 * h1).sum(dim=1, keepdim=True)

        # Concatenate
        mlp_in = torch.cat([h0, h1, t_emb, dist, dot_prod], dim=1)

        # --- 4. Prediction ---
        return self.mlp(mlp_in)
