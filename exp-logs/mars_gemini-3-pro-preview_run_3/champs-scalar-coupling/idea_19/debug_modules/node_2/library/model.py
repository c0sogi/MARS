import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius_graph
from torch_scatter import scatter_add
from library.config import Config


class GaussianSmearing(nn.Module):
    """
    Expands scalar distances into a vector of Gaussian Radial Basis Function (RBF) values.
    This provides a continuous, high-dimensional representation of distance.
    """

    def __init__(self, start: float = 0.0, stop: float = 5.0, num_gaussians: int = 50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        # The width (gamma) is determined by the spacing between centers
        self.coeff = -0.5 / ((stop - start) / (num_gaussians - 1)) ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        # dist: [num_edges] -> [num_edges, 1]
        # offset: [num_gaussians] -> [1, num_gaussians]
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class InteractionBlock(nn.Module):
    """
    A single interaction block of the Continuous Filter Network.
    Performs message passing where edge weights are generated dynamically
    from inter-atomic distances via a filter-generating MLP.
    """

    def __init__(self, hidden_channels: int, num_rbf: int):
        super().__init__()
        # Filter Generator Network: RBF -> Filter Weights
        self.mlp = nn.Sequential(
            nn.Linear(num_rbf, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # Transformation for incoming messages
        # Note: SchNet typically applies the filter element-wise to the neighbor's features.
        # Here we follow the standard formulation: x_j * W_ij

        # Output linear transformation (Update step)
        self.lin = nn.Linear(hidden_channels, hidden_channels)

        # Layer Norm for training stability in deeper networks
        self.norm = nn.LayerNorm(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.xavier_uniform_(self.mlp[2].weight)
        nn.init.xavier_uniform_(self.lin.weight)
        self.mlp[0].bias.data.fill_(0)
        self.mlp[2].bias.data.fill_(0)
        self.lin.bias.data.fill_(0)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [N, F]
            edge_index: Graph connectivity [2, E]
            edge_weight: RBF expanded distances [E, num_rbf]
        """
        # 1. Generate continuous filters from distances
        # W shape: [E, F]
        W = self.mlp(edge_weight)

        # 2. Message Passing
        # Gather source node features: x_j
        row, col = edge_index
        x_j = x[col]  # [E, F]

        # Apply filter (Element-wise multiplication)
        m = x_j * W  # [E, F]

        # 3. Aggregation (Sum pooling)
        # Aggregate messages to target nodes (row indices)
        aggr = scatter_add(m, row, dim=0, dim_size=x.size(0))  # [N, F]

        # 4. Update & Residual
        out = self.lin(aggr)
        out = self.norm(out)

        return x + out  # Residual connection


class MPDIN(nn.Module):
    """
    Molecule-Parallel Deep Interaction Network.

    Backbone: Node-Centric Continuous Filter Network (SchNet-variant).
    Head: Interaction-Aware Shared Conditional MLP.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.node_dim = config.node_dim
        self.num_rbf = config.num_rbf
        self.cutoff = config.cutoff
        self.type_embed_dim = config.type_embed_dim

        # 1. Atom Embedding (H, C, N, O, F -> 0..4)
        self.embedding = nn.Embedding(5, self.node_dim)

        # 2. Distance Expansion (RBF)
        self.distance_expansion = GaussianSmearing(
            start=config.rbf_min,
            stop=config.rbf_max,
            num_gaussians=self.num_rbf,
        )

        # 3. Interaction Blocks (Backbone)
        self.interactions = nn.ModuleList(
            [
                InteractionBlock(self.node_dim, self.num_rbf)
                for _ in range(config.num_layers)
            ]
        )

        # 4. Readout Head
        # Coupling Type Embedding (8 types)
        self.type_embedding = nn.Embedding(8, self.type_embed_dim)

        # Input dimension for the final MLP:
        # h_i (node_dim) + h_j (node_dim) + dot_prod (1) + dist_rbf (num_rbf) + type_emb (type_embed_dim)
        head_in_dim = self.node_dim * 2 + 1 + self.num_rbf + self.type_embed_dim

        self.head = nn.Sequential(
            nn.Linear(head_in_dim, self.node_dim),
            nn.SiLU(),
            nn.Linear(self.node_dim, self.node_dim // 2),
            nn.SiLU(),
            nn.Linear(self.node_dim // 2, 1),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: Dictionary containing collated molecule data.
                   Keys: atom_types, atom_coords, batch_index,
                         coupling_atom_index_0, coupling_atom_index_1, coupling_type
        Returns:
            pred: Predicted scalar coupling constants (standardized) [num_couplings, 1]
        """
        # Unpack batch data
        atom_types = batch["atom_types"]
        atom_coords = batch["atom_coords"]
        batch_index = batch["batch_index"]

        # --- 1. Graph Construction & Feature Initialization ---

        # Initialize node embeddings
        h = self.embedding(atom_types)  # [N, node_dim]

        # Construct Radius Graph dynamically
        # This creates edges between atoms within 'cutoff' distance, respecting batch boundaries
        edge_index = radius_graph(
            atom_coords, r=self.cutoff, batch=batch_index, max_num_neighbors=200
        )
        row, col = edge_index

        # Compute distances for graph edges
        dist = (atom_coords[row] - atom_coords[col]).norm(dim=-1)

        # Expand distances using RBF
        edge_rbf = self.distance_expansion(dist)  # [E, num_rbf]

        # --- 2. Backbone: Interaction Blocks ---

        for interaction in self.interactions:
            h = interaction(h, edge_index, edge_rbf)

        # --- 3. Readout Phase ---

        # Retrieve indices for the target coupling pairs
        idx0 = batch["coupling_atom_index_0"]
        idx1 = batch["coupling_atom_index_1"]
        c_types = batch["coupling_type"]

        # Gather node features for the pairs
        h0 = h[idx0]  # [num_couplings, node_dim]
        h1 = h[idx1]  # [num_couplings, node_dim]

        # Compute specific distances for the coupling pairs
        # (These pairs are targets, so we compute their exact distance regardless of graph cutoff)
        dist_coupling = (atom_coords[idx0] - atom_coords[idx1]).norm(dim=-1)
        rbf_coupling = self.distance_expansion(
            dist_coupling
        )  # [num_couplings, num_rbf]

        # Explicit Multiplicative Interaction (Scalar Dot Product)
        # Captures the alignment/interaction magnitude between the learned atom states
        dot_prod = (h0 * h1).sum(dim=-1, keepdim=True)  # [num_couplings, 1]

        # Coupling Type Embedding
        type_emb = self.type_embedding(c_types)  # [num_couplings, type_embed_dim]

        # Concatenate all features for the final prediction
        out = torch.cat([h0, h1, dot_prod, rbf_coupling, type_emb], dim=-1)

        # Predict
        pred = self.head(out)  # [num_couplings, 1]

        return pred
