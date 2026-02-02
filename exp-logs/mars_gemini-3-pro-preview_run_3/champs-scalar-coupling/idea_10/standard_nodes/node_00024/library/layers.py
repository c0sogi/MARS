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
    Performs node-centric continuous filter convolution (cf. SchNet).
    Aggregates messages from neighbors weighted by RBF distance filters.
    Optimized for speed to enable training on full dataset (Cite Lesson 00014).
    """

    def __init__(self, hidden_dim, num_rbf, activation=nn.SiLU()):
        super(InteractionBlock, self).__init__()
        self.hidden_dim = hidden_dim

        # Filter Network: Generates weights from edge embeddings
        self.filter_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
        )

        # MLP to process node features
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, edge_index, edge_embeddings):
        """
        Args:
            h: (num_nodes, hidden_dim)
            edge_index: (2, num_edges)
            edge_embeddings: (num_edges, hidden_dim) - Derived from RBF distances
        Returns:
            Updated h: (num_nodes, hidden_dim)
        """
        # 1. Compute Filters
        W = self.filter_network(edge_embeddings)  # (M, hidden_dim)

        # 2. Gather neighbor features
        src, dst = edge_index
        h_neighbors = h[src]  # (M, hidden_dim)

        # 3. Interaction
        messages = h_neighbors * W  # (M, hidden_dim)

        # 4. Aggregate
        aggregated = scatter(messages, dst, dim=0, dim_size=h.size(0), reduce="sum")

        # 5. Update Nodes (Residual)
        update = self.node_mlp(aggregated)
        return h + update


class SharedCouplingHead(nn.Module):
    """
    Shared readout module conditioned on coupling type embeddings.
    More data-efficient than disjoint heads (Cite Lesson 00022).
    Explicitly injects edge features (Cite Lesson 00011).
    """

    def __init__(
        self,
        node_dim,
        edge_dim,
        num_types=Config.NUM_COUPLING_TYPES,
        hidden_dim=128,
        output_dim=1,
    ):
        super(SharedCouplingHead, self).__init__()

        # Type embedding
        self.type_embedding = nn.Embedding(num_types, hidden_dim)

        # Input: Node_i + Node_j + Edge_ij + Type_Emb
        input_dim = node_dim * 2 + edge_dim + hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        node_embeddings,
        edge_embeddings,
        edge_index,
        coupling_edge_index,
        coupling_type,
    ):
        # 1. Gather features
        e_ij = edge_embeddings[coupling_edge_index]  # (C, edge_dim)

        src_idx = edge_index[0, coupling_edge_index]
        dst_idx = edge_index[1, coupling_edge_index]

        h_src = node_embeddings[src_idx]  # (C, node_dim)
        h_dst = node_embeddings[dst_idx]  # (C, node_dim)

        # Type embedding
        t_emb = self.type_embedding(coupling_type)  # (C, hidden_dim)

        # Combine
        combined = torch.cat([h_src, h_dst, e_ij, t_emb], dim=-1)

        # 2. Predict
        return self.mlp(combined)
