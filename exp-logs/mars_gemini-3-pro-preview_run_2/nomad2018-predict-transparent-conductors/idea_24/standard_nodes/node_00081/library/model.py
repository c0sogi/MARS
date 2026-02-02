import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from library.config import Config


class GaussianRBF(nn.Module):
    """
    Gaussian Radial Basis Function expansion for edge distances.
    """

    def __init__(self, start=0.0, cutoff=8.0, n_gaussians=60):
        super().__init__()
        self.cutoff = cutoff
        # Centers are linearly spaced
        offset = torch.linspace(start, cutoff, n_gaussians)
        # Widths (beta) are derived from the spacing
        width = offset[1] - offset[0]
        self.register_buffer("offset", offset)
        self.register_buffer("width", torch.tensor(width))

    def forward(self, dist):
        """
        Args:
            dist: Tensor of shape (num_edges, 1) or (num_edges,) containing distances.
        Returns:
            Tensor of shape (num_edges, n_gaussians)
        """
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(-1.0 * (dist / self.width).pow(2))


class ReceiverAwareGatedConv(MessagePassing):
    """
    Receiver-Aware Gated Convolution Layer.
    Generates messages based on Source, Target, and Edge features using a gated mechanism.
    Cite solution_lesson_node_00080: Gated Convolutions vs. Generic MLP
    Cite solution_lesson_node_00072: Receiver-Aware Message Passing
    """

    def __init__(self, hidden_dim):
        super().__init__(aggr="add")  # Sum aggregation

        # Input: [h_i || h_j || e_ij] -> 3 * hidden_dim
        # Output: 2 * hidden_dim (Filter + Gate)
        self.linear = nn.Linear(3 * hidden_dim, 2 * hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        # x: (num_nodes, hidden_dim)
        # edge_index: (2, num_edges)
        # edge_attr: (num_edges, hidden_dim) -> Projected RBF features
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i: Target node features (num_edges, hidden_dim)
        # x_j: Source node features (num_edges, hidden_dim)
        # edge_attr: Edge features (num_edges, hidden_dim)

        # Concatenate Source, Target, and Edge
        z_ij = torch.cat([x_i, x_j, edge_attr], dim=-1)

        # Linear transformation
        out = self.linear(z_ij)

        # Split into filter and gate
        filter_part, gate_part = out.chunk(2, dim=-1)

        # Gated activation: Softplus(filter) * Sigmoid(gate)
        return F.softplus(filter_part) * torch.sigmoid(gate_part)


class InteractionBlock(nn.Module):
    """
    Interaction Block with Learnable Scalar Residual and BatchNorm.
    """

    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.conv = ReceiverAwareGatedConv(hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Learnable scalar epsilon, initialized to 0
        self.epsilon = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index, edge_attr):
        # Message Passing
        # Agg(m_ij)
        aggr_out = self.conv(x, edge_index, edge_attr)

        # Residual connection with learnable scalar
        # (1 + epsilon) * h_l
        residual = (1.0 + self.epsilon) * x

        # Combine
        out = aggr_out + residual

        # BatchNorm
        out = self.bn(out)

        # Activation (Softplus as per description)
        out = F.softplus(out)

        # Dropout
        out = self.dropout(out)

        return out


class kRACGN(nn.Module):
    """
    k-Nearest Neighbor Receiver-Aware Crystal Graph Network.
    """

    def __init__(self):
        super().__init__()

        # Hyperparameters from Config
        hidden_dim = Config.HIDDEN_DIM
        num_blocks = Config.NUM_BLOCKS
        dropout = Config.DROPOUT
        rbf_bins = Config.RBF_BINS
        rbf_cutoff = Config.RBF_CUTOFF

        # 1. Node Embedding
        # Embedding atomic numbers (1 to ~100). 118 is max element.
        self.node_embedding = nn.Embedding(119, hidden_dim)

        # 2. Edge Features & Shared Projection
        self.rbf = GaussianRBF(start=0.0, cutoff=rbf_cutoff, n_gaussians=rbf_bins)
        self.edge_projection = nn.Linear(rbf_bins, hidden_dim)

        # 3. Interaction Blocks
        self.blocks = nn.ModuleList(
            [InteractionBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )

        # 4. Decoupled Readout Heads
        # Formation Energy Head
        self.head_formation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

        # Bandgap Energy Head
        self.head_bandgap = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        """
        Forward pass.
        Args:
            data: torch_geometric.data.Data object containing:
                - x: Atomic numbers (num_nodes,)
                - edge_index: Graph connectivity (2, num_edges)
                - edge_attr: Interatomic distances (num_edges, 1)
                - batch: Batch vector (num_nodes,)
        """
        x, edge_index, edge_dist, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # 1. Initialize Node Embeddings
        h = self.node_embedding(x)  # (num_nodes, hidden_dim)

        # 2. Process Edge Features (Shared Projection)
        # Expand distances using RBF
        edge_rbf = self.rbf(edge_dist)  # (num_edges, rbf_bins)
        # Project to hidden dimension
        e_ij = self.edge_projection(edge_rbf)  # (num_edges, hidden_dim)

        # 3. Interaction Blocks
        for block in self.blocks:
            h = block(h, edge_index, e_ij)

        # 4. Global Pooling
        # Global Mean Pooling is robust for intensive properties
        h_pool = global_mean_pool(h, batch)  # (batch_size, hidden_dim)

        # 5. Prediction Heads
        out_formation = self.head_formation(h_pool)
        out_bandgap = self.head_bandgap(h_pool)

        # Concatenate predictions (batch_size, 2)
        return torch.cat([out_formation, out_bandgap], dim=1)
