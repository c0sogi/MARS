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
)


class DWConvLayer(MessagePassing):
    """
    Distance-Weighted Graph Convolutional Layer.
    Update Rule: h_i = ReLU( W_self * h_i + W_neigh * sum( (1/d_ij^2) * h_j ) )
    """

    def __init__(self, in_channels, out_channels):
        # aggr='add' sums the messages from neighbors
        super(DWConvLayer, self).__init__(aggr="add")
        self.lin_self = nn.Linear(in_channels, out_channels)
        self.lin_neigh = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        """
        Args:
            x (Tensor): Node features of shape (N, in_channels)
            edge_index (LongTensor): Graph connectivity of shape (2, E)
            edge_weight (Tensor): Edge weights (1/d^2) of shape (E,)
        """
        # Propagate messages along edges
        # This calls message(), aggregate(), and update() internally
        aggr_out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        # Apply linear transformations and combine
        # Self-loop contribution + Neighbor contribution
        out = self.lin_self(x) + self.lin_neigh(aggr_out)

        return F.relu(out)

    def message(self, x_j, edge_weight):
        """
        Constructs the message from neighbor j to node i.
        Weighted by the edge attribute (inverse squared distance).
        """
        # x_j: features of neighbors, shape (E, in_channels)
        # edge_weight: shape (E,)
        return x_j * edge_weight.view(-1, 1)


class CouplingPredictor(nn.Module):
    """
    Main architecture for Scalar Coupling Prediction.
    Combines DW-GCN for structure encoding and an MLP for pair-wise regression.
    """

    def __init__(self):
        super(CouplingPredictor, self).__init__()

        # 1. Embeddings
        self.atom_embedding = nn.Embedding(NUM_ATOM_TYPES, EMBED_DIM)
        self.type_embedding = nn.Embedding(NUM_COUPLING_TYPES, EMBED_DIM)

        # 2. Graph Convolutional Layers
        self.gcn_layers = nn.ModuleList()

        # First layer maps from Embedding Dim to Hidden Dim
        self.gcn_layers.append(DWConvLayer(EMBED_DIM, HIDDEN_DIM))

        # Subsequent layers stay in Hidden Dim
        for _ in range(NUM_GCN_LAYERS - 1):
            self.gcn_layers.append(DWConvLayer(HIDDEN_DIM, HIDDEN_DIM))

        # 3. Prediction Head (MLP)
        # Input features:
        #   - Node 0 Feature (HIDDEN_DIM)
        #   - Node 1 Feature (HIDDEN_DIM)
        #   - Coupling Type Embedding (EMBED_DIM)
        #   - Distance (1)
        #   - Dot Product of Node Features (1)
        mlp_input_dim = (HIDDEN_DIM * 2) + EMBED_DIM + 1 + 1

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
        # Embed atoms
        h = self.atom_embedding(x)  # (TotalNodes, EMBED_DIM)

        # Apply GCN layers
        for layer in self.gcn_layers:
            h = layer(h, edge_index, edge_attr)

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

        # Distance
        if dist.dim() == 1:
            dist = dist.unsqueeze(1)  # (B, 1)

        # Dot Product (Interaction alignment)
        # (B, D) * (B, D) -> (B, D) -> sum -> (B, 1)
        dot_prod = (h0 * h1).sum(dim=1, keepdim=True)

        # Concatenate all features
        mlp_in = torch.cat([h0, h1, t_emb, dist, dot_prod], dim=1)

        # --- 4. Prediction ---
        out = self.mlp(mlp_in)

        return out
