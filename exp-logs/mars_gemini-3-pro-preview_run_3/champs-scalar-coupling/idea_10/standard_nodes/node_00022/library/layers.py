import torch
import torch.nn as nn
from torch_scatter import scatter
from library.config import Config
from library.utils import GaussianSmearing


class EdgeEmbedding(nn.Module):
    """
    Initializes edge features from RBF-expanded distances.
    Projects scalar distances into a high-dimensional vector space.
    """

    def __init__(
        self, num_rbf, hidden_dim, cutoff_lower=0.0, cutoff_upper=Config.CUTOFF_RADIUS
    ):
        super(EdgeEmbedding, self).__init__()
        self.rbf = GaussianSmearing(
            start=cutoff_lower,
            stop=cutoff_upper,
            num_gaussians=num_rbf,
            centered=False,
            learnable=False,
        )
        self.mlp = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, edge_attr):
        """
        Args:
            edge_attr: Tensor of shape (num_edges,) containing scalar distances.
        Returns:
            Tensor of shape (num_edges, hidden_dim)
        """
        # Expand distances: (num_edges, num_rbf)
        rbf_features = self.rbf(edge_attr)
        # Project to hidden dim
        return self.mlp(rbf_features)


class InteractionBlock(nn.Module):
    """
    Performs directional message passing.
    Aggregates messages from 'preceding' edges (k->j) to 'current' edges (j->i)
    weighted by angular information derived from the triplet (k, j, i).
    """

    def __init__(self, hidden_dim, num_angle_rbf, activation=nn.SiLU()):
        super(InteractionBlock, self).__init__()
        self.hidden_dim = hidden_dim

        # Angular RBF expansion for cosine theta (range -1 to 1)
        self.angle_rbf = GaussianSmearing(
            start=-1.0,
            stop=1.0,
            num_gaussians=num_angle_rbf,
            centered=True,
            learnable=False,
        )

        # MLP to transform angular features
        self.angle_mlp = nn.Sequential(
            nn.Linear(num_angle_rbf, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Linear transform for source edge features (k->j)
        self.source_lin = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # MLP to update target edge features (j->i) using aggregated messages
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, edge_embeddings, triplet_index, triplet_attr):
        """
        Args:
            edge_embeddings: (num_edges, hidden_dim)
            triplet_index: (2, num_triplets). Row 0: source edge indices (k->j),
                           Row 1: target edge indices (j->i).
            triplet_attr: (num_triplets,) cosine of bond angles theta_{kji}.
        Returns:
            Updated edge_embeddings: (num_edges, hidden_dim)
        """
        # 1. Process Angular Features
        angle_emb = self.angle_rbf(triplet_attr)  # (num_triplets, num_angle_rbf)
        angle_w = self.angle_mlp(angle_emb)  # (num_triplets, hidden_dim)

        # 2. Process Source Edges
        source_idx = triplet_index[0]
        target_idx = triplet_index[1]

        source_h = edge_embeddings[source_idx]  # (num_triplets, hidden_dim)
        source_h = self.source_lin(source_h)  # (num_triplets, hidden_dim)

        # 3. Compute Messages (Hadamard product of source edge and angle weight)
        messages = source_h * angle_w  # (num_triplets, hidden_dim)

        # 4. Aggregate Messages to Target Edges
        # Sum messages for each target edge index
        num_edges = edge_embeddings.size(0)
        aggregated_messages = scatter(
            messages, target_idx, dim=0, dim_size=num_edges, reduce="sum"
        )

        # 5. Update Target Edges
        # Concatenate original features with aggregated messages (Residual-like structure)
        combined = torch.cat([edge_embeddings, aggregated_messages], dim=-1)
        update = self.update_mlp(combined)

        # Residual connection
        return edge_embeddings + update


class TypeSpecificHeads(nn.Module):
    """
    Readout module with specific heads for each coupling type.
    Injects edge embeddings and node embeddings into the final prediction to ensure
    high-frequency geometric information is available at the output.
    """

    def __init__(
        self,
        node_dim,
        edge_dim,
        num_types=Config.NUM_COUPLING_TYPES,
        hidden_dim=128,
        output_dim=1,
    ):
        super(TypeSpecificHeads, self).__init__()
        self.num_types = num_types

        # Input: Node_i + Node_j + Edge_ij
        input_dim = node_dim * 2 + edge_dim

        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, output_dim),
                )
                for _ in range(num_types)
            ]
        )

    def forward(
        self,
        node_embeddings,
        edge_embeddings,
        edge_index,
        coupling_edge_index,
        coupling_type,
    ):
        """
        Args:
            node_embeddings: (num_nodes, node_dim)
            edge_embeddings: (num_edges, edge_dim)
            edge_index: (2, num_edges) - Global connectivity
            coupling_edge_index: (num_couplings,) indices of edges corresponding to the target pairs
            coupling_type: (num_couplings,) integer type indices
        Returns:
            predictions: (num_couplings, output_dim)
        """
        # 1. Gather features for the specific couplings

        # Edge features for the coupling pair (e_ij)
        e_ij = edge_embeddings[coupling_edge_index]  # (num_couplings, edge_dim)

        # Identify source and target nodes for these edges
        # edge_index[0] is source, edge_index[1] is target
        src_idx = edge_index[0, coupling_edge_index]
        dst_idx = edge_index[1, coupling_edge_index]

        # Gather node features (h_i, h_j)
        h_src = node_embeddings[src_idx]  # (num_couplings, node_dim)
        h_dst = node_embeddings[dst_idx]  # (num_couplings, node_dim)

        # Combine all features
        combined = torch.cat([h_src, h_dst, e_ij], dim=-1)  # (num_couplings, input_dim)

        # 2. Apply Type-Specific Heads
        # Initialize output tensor
        output = torch.zeros(combined.size(0), 1, device=combined.device)

        # Iterate over types and apply the corresponding MLP head
        for t in range(self.num_types):
            mask = coupling_type == t
            if mask.any():
                # Process subset of couplings belonging to type t
                subset_input = combined[mask]
                subset_out = self.heads[t](subset_input)
                output[mask] = subset_out

        return output
