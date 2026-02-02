import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from library.config import (
    HIDDEN_DIM,
    NUM_BLOCKS,
    DROPOUT_RATE,
    ATOM_EMBEDDING_DIM,
    EDGE_EMBEDDING_DIM,
    NUM_RBF_BINS,
    CUTOFF_RADIUS,
    RBF_GAMMA,
)


class GaussianRBF(nn.Module):
    """
    Expands scalar distances into a vector of Gaussian radial basis functions.
    """

    def __init__(
        self, start=0.0, stop=CUTOFF_RADIUS, num_bins=NUM_RBF_BINS, gamma=RBF_GAMMA
    ):
        super().__init__()
        self.centers = nn.Parameter(
            torch.linspace(start, stop, num_bins), requires_grad=False
        )
        self.gamma = gamma

    def forward(self, x):
        # x: (E, 1) distances
        # centers: (num_bins, )
        # output: (E, num_bins)
        return torch.exp(-self.gamma * (x - self.centers) ** 2)


class ReceiverAwareInteractionBlock(MessagePassing):
    """
    Monolithic Receiver-Aware Interaction Block with Gated Convolution.
    Concatenates source, target, and edge features.
    Cite solution_lesson_node_00084: Monolithic outperforms Multi-Head on small data.
    Cite solution_lesson_node_00072: Receiver-Awareness improves performance.
    """

    def __init__(self, hidden_dim, dropout):
        super().__init__(aggr="add")
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Monolithic Linear layers for Filter and Gate
        # Input: h_i || h_j || e_ij -> 3 * hidden_dim
        self.lin_filter = nn.Linear(3 * hidden_dim, hidden_dim)
        self.lin_gate = nn.Linear(3 * hidden_dim, hidden_dim)

        # Update function
        self.lin_update = nn.Linear(hidden_dim, hidden_dim)

        # Batch Normalization
        self.bn = nn.BatchNorm1d(hidden_dim)

        # Learnable residual scalar initialized to 0
        # Cite solution_lesson_node_00060, solution_lesson_node_00065
        self.epsilon = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, edge_index, edge_attr):
        # x: (N, hidden_dim)
        # edge_index: (2, E)
        # edge_attr: (E, hidden_dim) - projected edge features

        # Save input for residual connection
        x_in = x

        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Update
        out = self.lin_update(out)
        out = F.dropout(out, p=self.dropout, training=self.training)

        # Residual connection with learnable scalar
        # h_{l+1} = Softplus(BN(Agg + (1 + eps) * h_l))
        out = out + (1.0 + self.epsilon) * x_in
        out = self.bn(out)
        out = F.softplus(out)  # Cite solution_lesson_node_00041

        return out

    def message(self, x_i, x_j, edge_attr):
        # Concatenate source, target, edge
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)

        # Gated Convolution
        # Cite solution_lesson_node_00080: Gated Convolutions prefered.
        f = F.softplus(self.lin_filter(z))
        g = torch.sigmoid(self.lin_gate(z))
        return f * g


class CrystalGraphNetwork(nn.Module):
    """
    Monolithic Receiver-Aware Crystal Graph Network.
    """

    def __init__(self):
        super().__init__()

        # Node Embedding (Atomic numbers up to 118)
        self.atom_embedding = nn.Embedding(119, HIDDEN_DIM)

        # Edge Embedding and Projection
        self.rbf = GaussianRBF()
        self.edge_proj = nn.Linear(NUM_RBF_BINS, HIDDEN_DIM)

        # Stack of Interaction Blocks
        self.blocks = nn.ModuleList(
            [
                ReceiverAwareInteractionBlock(HIDDEN_DIM, DROPOUT_RATE)
                for _ in range(NUM_BLOCKS)
            ]
        )

        # Decoupled Prediction Heads
        # Head for Formation Energy
        self.head_formation = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.Softplus(), nn.Linear(HIDDEN_DIM, 1)
        )

        # Head for Bandgap Energy
        self.head_bandgap = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.Softplus(), nn.Linear(HIDDEN_DIM, 1)
        )

    def forward(self, data):
        # Unpack data
        x = data.x.squeeze()  # (N, )
        edge_index = data.edge_index
        edge_dist = data.edge_attr  # (E, 1)
        batch = data.batch

        # 1. Embed Nodes
        h = self.atom_embedding(x)  # (N, hidden_dim)

        # 2. Embed and Project Edges
        rbf_feat = self.rbf(edge_dist)  # (E, num_bins)
        e = self.edge_proj(rbf_feat)  # (E, hidden_dim)

        # 3. Apply Interaction Blocks
        for block in self.blocks:
            h = block(h, edge_index, e)

        # 4. Global Mean Pooling
        h_pool = global_mean_pool(h, batch)  # (Batch, hidden_dim)

        # 5. Prediction Heads
        pred_formation = self.head_formation(h_pool)
        pred_bandgap = self.head_bandgap(h_pool)

        # Concatenate results (Batch, 2)
        return torch.cat([pred_formation, pred_bandgap], dim=1)
