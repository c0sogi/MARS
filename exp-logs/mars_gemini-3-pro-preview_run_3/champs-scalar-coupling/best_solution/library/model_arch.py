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
    RBF_MIN,
    RBF_MAX,
    NUM_RBF,
)


class RBFExpansion(nn.Module):
    """
    Expands a distance scalar into a vector of Gaussian Basis Functions.
    """

    def __init__(self, min_dist=RBF_MIN, max_dist=RBF_MAX, num_rbf=NUM_RBF):
        super().__init__()
        # Register centers as buffer (not a learnable parameter)
        centers = torch.linspace(min_dist, max_dist, num_rbf)
        self.register_buffer("centers", centers)

        # Width of the Gaussians
        self.sigma = (max_dist - min_dist) / num_rbf
        self.gamma = 1.0 / (self.sigma**2)

    def forward(self, dist):
        """
        Args:
            dist (Tensor): Shape (E,) or (E, 1)
        Returns:
            Tensor: Shape (E, num_rbf)
        """
        dist = dist.view(-1, 1)
        return torch.exp(-self.gamma * (dist - self.centers) ** 2)


class InteractionLayer(MessagePassing):
    """
    Continuous Filter Convolution Layer (similar to SchNet interaction block).
    Uses RBF expanded distances to generate edge-specific filters.
    """

    def __init__(self, hidden_dim, num_rbf):
        super(InteractionLayer, self).__init__(aggr="add")

        # Filter generator: Maps RBF expansion to Hidden Dim
        self.mlp_filter = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Update network
        self.lin_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr_rbf):
        """
        Args:
            x (Tensor): Node features (N, hidden_dim)
            edge_index (LongTensor): (2, E)
            edge_attr_rbf (Tensor): Expanded edge features (E, num_rbf)
        """
        # Generate filters from edge features
        W = self.mlp_filter(edge_attr_rbf)  # (E, hidden_dim)

        # Propagate
        aggr_out = self.propagate(edge_index, x=x, W=W)

        # Residual update
        return x + self.lin_update(aggr_out)

    def message(self, x_j, W):
        """
        Element-wise product of neighbor features and generated filter.
        """
        return x_j * W


class CouplingPredictor(nn.Module):
    """
    Main architecture for Scalar Coupling Prediction.
    Uses Continuous Filter Convolutions (Interaction Layers) with RBF expansion.
    """

    def __init__(self):
        super(CouplingPredictor, self).__init__()

        # 1. Embeddings & RBF
        self.atom_embedding = nn.Embedding(NUM_ATOM_TYPES, HIDDEN_DIM)
        self.type_embedding = nn.Embedding(NUM_COUPLING_TYPES, HIDDEN_DIM)
        self.rbf = RBFExpansion(RBF_MIN, RBF_MAX, NUM_RBF)

        # 2. Interaction Layers
        self.layers = nn.ModuleList()
        for _ in range(NUM_GCN_LAYERS):
            self.layers.append(InteractionLayer(HIDDEN_DIM, NUM_RBF))

        # 3. Prediction Head (MLP)
        # Input features:
        #   - Node 0 Feature (HIDDEN_DIM)
        #   - Node 1 Feature (HIDDEN_DIM)
        #   - Coupling Type Embedding (HIDDEN_DIM)
        #   - Distance RBF (NUM_RBF) - Expanded distance provides more info than scalar
        #   - Dot Product (1)
        mlp_input_dim = (HIDDEN_DIM * 3) + NUM_RBF + 1

        self.mlp = nn.Sequential()
        current_dim = mlp_input_dim

        for i in range(NUM_MLP_LAYERS - 1):
            self.mlp.add_module(f"lin_{i}", nn.Linear(current_dim, HIDDEN_DIM))
            self.mlp.add_module(f"bn_{i}", nn.BatchNorm1d(HIDDEN_DIM))
            self.mlp.add_module(f"relu_{i}", nn.ReLU())
            self.mlp.add_module(f"drop_{i}", nn.Dropout(DROPOUT))
            current_dim = HIDDEN_DIM

        # Final regression layer
        self.mlp.add_module("lin_out", nn.Linear(current_dim, 1))

    def forward(self, data):
        """
        Args:
            data (Batch): PyG Batch object containing graph and task data.
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        target_pair = data.target_pair  # Shape (B, 2), local indices
        type_idx = data.type_idx  # Shape (B,) or (B, 1)
        dist = data.dist  # Shape (B,) or (B, 1)

        # --- 1. Graph Encoding ---
        # Expand edge distances
        edge_attr_rbf = self.rbf(edge_attr)

        # Embed atoms
        h = self.atom_embedding(x)  # (TotalNodes, HIDDEN_DIM)

        # Apply Interaction layers
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr_rbf)

        # --- 2. Pair-Centric Readout ---
        # We need to extract the features for the specific atom pairs.
        # target_pair contains indices local to each molecule (0..NumAtoms-1).
        # We need to convert these to global indices in the batched 'h' tensor.

        if hasattr(data, "ptr"):
            # data.ptr is a tensor of offsets [0, N1, N1+N2, ...]
            # We take the start index for each graph in the batch
            batch_offsets = data.ptr[:-1].to(target_pair.device)
        else:
            # Fallback: compute offsets from batch vector
            # This handles cases where ptr is missing (though rare in PyG DataLoaders)
            unique, counts = torch.unique(data.batch, sorted=True, return_counts=True)
            batch_offsets = torch.cat(
                [
                    torch.tensor([0], device=target_pair.device),
                    torch.cumsum(counts, dim=0)[:-1],
                ]
            )

        # Broadcast offsets to match target_pair shape (B, 2)
        # global_idx = local_idx + graph_start_idx
        global_pair_indices = target_pair + batch_offsets.view(-1, 1)

        # Gather node features
        idx_0 = global_pair_indices[:, 0]
        idx_1 = global_pair_indices[:, 1]

        h0 = h[idx_0]  # (B, HIDDEN_DIM)
        h1 = h[idx_1]  # (B, HIDDEN_DIM)

        # --- 3. Feature Fusion ---
        # Ensure shapes are correct for concatenation

        # Coupling Type Embedding
        t_emb = self.type_embedding(type_idx)
        if t_emb.dim() == 3:  # Handle (B, 1, D)
            t_emb = t_emb.squeeze(1)

        # Distance Expansion
        dist_rbf = self.rbf(dist)  # (B, NUM_RBF)
        if dist_rbf.dim() == 3:
            dist_rbf = dist_rbf.squeeze(1)

        # Dot Product (Interaction alignment)
        # (B, D) * (B, D) -> (B, D) -> sum -> (B, 1)
        dot_prod = (h0 * h1).sum(dim=1, keepdim=True)

        # Concatenate all features
        # h0, h1, t_emb are (B, HIDDEN_DIM)
        # dist_rbf is (B, NUM_RBF)
        # dot_prod is (B, 1)
        mlp_in = torch.cat([h0, h1, t_emb, dist_rbf, dot_prod], dim=1)

        # --- 4. Prediction ---
        out = self.mlp(mlp_in)

        return out
