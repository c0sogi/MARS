import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from library.config import (
    HIDDEN_DIM,
    NUM_HEADS,
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


class MultiHeadInteractionBlock(MessagePassing):
    """
    Interaction block with Multi-Head Gating and Receiver-Awareness.
    Splits features into heads, computes independent gated messages, and aggregates them.
    """

    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__(aggr="add")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout

        assert hidden_dim % num_heads == 0, "Hidden dim must be divisible by num_heads"

        # Head-specific layers
        # Input to each head is concatenation of source (head_dim), target (head_dim), and edge (head_dim)
        # Total input dim per head = 3 * head_dim
        self.head_filters = nn.ModuleList(
            [nn.Linear(3 * self.head_dim, self.head_dim) for _ in range(num_heads)]
        )
        self.head_gates = nn.ModuleList(
            [nn.Linear(3 * self.head_dim, self.head_dim) for _ in range(num_heads)]
        )

        # Mixing layer to combine information from heads
        self.mix_linear = nn.Linear(hidden_dim, hidden_dim)

        # Batch Normalization
        self.bn = nn.BatchNorm1d(hidden_dim)

        # Learnable residual scalar initialized to 0
        self.epsilon = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, edge_index, edge_attr):
        # x: (N, hidden_dim)
        # edge_index: (2, E)
        # edge_attr: (E, hidden_dim) - projected edge features

        # Save input for residual connection
        x_in = x

        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Mix heads
        out = self.mix_linear(out)
        out = F.dropout(out, p=self.dropout, training=self.training)

        # Residual connection with learnable scalar
        # h_{l+1} = Softplus(BN(Agg + (1 + eps) * h_l))
        out = out + (1.0 + self.epsilon) * x_in
        out = self.bn(out)
        out = F.softplus(out)

        return out

    def message(self, x_i, x_j, edge_attr):
        # x_i, x_j: (E, hidden_dim)
        # edge_attr: (E, hidden_dim)

        # Split features into heads
        # Reshape to (E, num_heads, head_dim)
        x_i_split = x_i.view(-1, self.num_heads, self.head_dim)
        x_j_split = x_j.view(-1, self.num_heads, self.head_dim)
        edge_split = edge_attr.view(-1, self.num_heads, self.head_dim)

        head_messages = []

        for k in range(self.num_heads):
            # Receiver-Aware Concatenation for head k
            # z_k = [h_i^{(k)} || h_j^{(k)} || e_{ij}^{(k)}]
            # Shape: (E, 3 * head_dim)
            z_k = torch.cat(
                [x_i_split[:, k, :], x_j_split[:, k, :], edge_split[:, k, :]], dim=-1
            )

            # Filter: Softplus(W_f * z_k)
            f_k = F.softplus(self.head_filters[k](z_k))

            # Gate: Sigmoid(W_g * z_k)
            g_k = torch.sigmoid(self.head_gates[k](z_k))

            # Message: f_k * g_k
            m_k = f_k * g_k
            head_messages.append(m_k)

        # Concatenate messages from all heads back to (E, hidden_dim)
        return torch.cat(head_messages, dim=-1)


class MH_RA_CGN(nn.Module):
    """
    Multi-Head Receiver-Aware Crystal Graph Network.
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
                MultiHeadInteractionBlock(HIDDEN_DIM, NUM_HEADS, DROPOUT_RATE)
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
