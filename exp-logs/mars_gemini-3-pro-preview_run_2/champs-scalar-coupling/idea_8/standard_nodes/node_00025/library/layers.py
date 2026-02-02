import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.utils import to_dense_batch
from library.config import Config


class InteractionBlock(nn.Module):
    """
    Directional Message Passing Layer (DMPNN) with Geometric Features.
    Operates on directed edges and updates their embeddings based on:
    1. Incoming edges (k->j)
    2. Triplet angles (k-j-i) via SBF
    3. Radial distance (j-i) via RBF
    """

    def __init__(self, hidden_dim, num_rbf, num_sbf, activation=F.silu):
        super(InteractionBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.activation = activation

        # Transformation for incoming edge features (k->j)
        self.lin_kj = nn.Linear(hidden_dim, hidden_dim)

        # Transformation for Spherical Basis Functions (Angle k-j-i)
        self.lin_sbf = nn.Linear(num_sbf, hidden_dim, bias=False)

        # Transformation for Radial Basis Functions (Distance j-i)
        self.lin_rbf = nn.Linear(num_rbf, hidden_dim, bias=False)

        # Deep MLP for the update step
        # Input: (h_old + m_agg + rbf_proj) -> Output: h_update
        # We explicitly avoid reducing these to linear projections to prevent information bottlenecks.
        self.mlp_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, rbf, sbf, triplet_indices):
        """
        Args:
            h (Tensor): Edge embeddings (num_edges, hidden_dim).
            rbf (Tensor): RBF features for edges (num_edges, num_rbf).
            sbf (Tensor): SBF features for triplets (num_triplets, num_sbf).
            triplet_indices (Tensor): (2, num_triplets) containing [edge_kj_idx, edge_ji_idx].

        Returns:
            Tensor: Updated edge embeddings (num_edges, hidden_dim).
        """
        # Unpack indices
        idx_kj, idx_ji = triplet_indices
        num_edges = h.size(0)

        # 1. Transform incoming edges (k->j)
        x_kj = self.lin_kj(h)  # (num_edges, hidden_dim)

        # 2. Gather incoming edges for triplets
        # We only care about edges that are part of a valid triplet
        x_kj_triplet = x_kj[idx_kj]  # (num_triplets, hidden_dim)

        # 3. Transform SBF (angular info)
        w_sbf = self.lin_sbf(sbf)  # (num_triplets, hidden_dim)

        # 4. Interaction: Hadamard product of incoming edge and angle info
        # This encodes the angular dependency of the interaction
        m_triplet = x_kj_triplet * w_sbf

        # 5. Aggregate messages to target edges (j->i)
        # Sum all m_triplet that map to the same target edge index (idx_ji)
        m_agg = scatter(m_triplet, idx_ji, dim=0, dim_size=num_edges, reduce="sum")

        # 6. Transform RBF (distance info) for the target edge
        w_rbf = self.lin_rbf(rbf)

        # 7. Update State
        # Combine: Previous state + Aggregated Message + Distance Info
        # We use addition (residual-like) before the MLP
        h_input = h + m_agg + w_rbf

        # Apply Deep MLP
        h_update = self.mlp_update(h_input)

        # Residual connection for the whole block
        return h + h_update


class GlobalAttentionBlock(nn.Module):
    """
    Graph Transformer Module for capturing long-range dependencies.
    Operates on atom-level representations to allow all-to-all communication.
    """

    def __init__(self, hidden_dim, num_heads, ff_dim, dropout=0.0):
        super(GlobalAttentionBlock, self).__init__()

        # Standard Transformer Encoder Layer
        # batch_first=True expects input (Batch, Seq, Feature)
        # norm_first=True (Pre-LN) is generally more stable for deep networks
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(self, x, batch):
        """
        Args:
            x (Tensor): Node features (num_nodes, hidden_dim).
            batch (Tensor): Batch indices for nodes (num_nodes,).

        Returns:
            Tensor: Updated node features (num_nodes, hidden_dim).
        """
        # 1. Dense Batching
        # Converts the sparse stacked graph batch into a dense tensor (B, N_max, D)
        # mask indicates valid nodes (True) vs padding (False)
        x_dense, mask = to_dense_batch(x, batch)

        # 2. Apply Transformer
        # src_key_padding_mask expects True for PADDED positions (ignored)
        # mask is True for VALID positions, so we invert it
        padding_mask = ~mask

        x_dense_out = self.transformer_layer(x_dense, src_key_padding_mask=padding_mask)

        # 3. Recover sparse format
        # Select only the valid nodes using the mask to return to (num_nodes, hidden_dim)
        x_out = x_dense_out[mask]

        return x_out


class ReadoutBlock(nn.Module):
    """
    Deterministic Fusion & Readout Module.
    Combines Global Node Embeddings and Local Edge Embeddings to predict coupling constant.
    """

    def __init__(self, hidden_dim, type_dim=8):
        super(ReadoutBlock, self).__init__()

        # Embedding for coupling type (e.g., 1JHC, 2JHH...)
        self.type_embedding = nn.Embedding(type_dim, type_dim)

        # Input dimension: Node_u (Global) + Node_v (Global) + Edge_uv (Local) + Type
        input_dim = (hidden_dim * 3) + type_dim

        # Deterministic MLP (No Dropout as per instructions)
        # Used to regress the scalar coupling constant
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, node_u, node_v, edge_uv, coupling_types):
        """
        Args:
            node_u (Tensor): Global embeddings for atom 0 (N_couplings, hidden_dim).
            node_v (Tensor): Global embeddings for atom 1 (N_couplings, hidden_dim).
            edge_uv (Tensor): Local edge embeddings for u->v (N_couplings, hidden_dim).
            coupling_types (Tensor): Integer coupling types (N_couplings,).

        Returns:
            Tensor: Predicted scalar coupling constants (N_couplings, 1).
        """
        # Get type embeddings
        type_emb = self.type_embedding(coupling_types)

        # Concatenate all features
        # [Global_u, Global_v, Local_Edge_uv, Type]
        cat_feat = torch.cat([node_u, node_v, edge_uv, type_emb], dim=1)

        # Predict
        out = self.mlp(cat_feat)

        return out
