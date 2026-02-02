import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from library.config import Config
from library.features import DualScaleRBF


class ReceiverAwareInteraction(MessagePassing):
    """
    Receiver-Aware Interaction Block.
    Constructs messages using concatenated receiver, sender, and edge features
    with Softplus/Sigmoid gating.
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super(ReceiverAwareInteraction, self).__init__(aggr="add")
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Input dimension: node_i + node_j + edge_attr
        # edge_attr comes from DualScaleRBF which projects to hidden_dim
        input_dim = hidden_dim * 3

        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(input_dim, hidden_dim)

        # Initialize weights
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        self.lin1.bias.data.fill_(0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr):
        # x: [num_nodes, hidden_dim]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, hidden_dim]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i: receiver [num_edges, hidden_dim]
        # x_j: sender [num_edges, hidden_dim]
        # edge_attr: edge embedding [num_edges, hidden_dim]

        # Concatenate features: z_ij = [h_i || h_j || e_ij]
        z_ij = torch.cat([x_i, x_j, edge_attr], dim=-1)

        # Gating Mechanism
        # m_ij = Softplus(Linear1(z_ij)) * Sigmoid(Linear2(z_ij))
        gate_1 = F.softplus(self.lin1(z_ij))
        gate_2 = torch.sigmoid(self.lin2(z_ij))

        m_ij = gate_1 * gate_2

        # Apply dropout to the message
        m_ij = F.dropout(m_ij, p=self.dropout, training=self.training)

        return m_ij


class AdaptiveResidualBlock(nn.Module):
    """
    Stabilized Adaptive Residual Block.
    Update rule: h_{l+1} = Softplus(BatchNorm(Agg(m_ij) + (1 + epsilon) * h_l))
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super(AdaptiveResidualBlock, self).__init__()
        self.interaction = ReceiverAwareInteraction(hidden_dim, dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

        # Learnable scalar epsilon, initialized to 0
        self.epsilon = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index, edge_attr):
        # Aggregated messages
        aggr_msg = self.interaction(x, edge_index, edge_attr)

        # Adaptive residual connection
        # (1 + epsilon) * h_l
        residual = (1.0 + self.epsilon) * x

        # Sum-Normalization
        out = aggr_msg + residual
        out = self.batch_norm(out)

        # Activation
        out = F.softplus(out)

        return out


class MS_RA_CGN(nn.Module):
    """
    Multi-Scale Receiver-Aware Crystal Graph Network.
    """

    def __init__(self):
        super(MS_RA_CGN, self).__init__()

        hidden_dim = Config.HIDDEN_DIM
        dropout = Config.DROPOUT

        # 1. Node Embedding
        self.atom_embedding = nn.Embedding(Config.MAX_ATOMIC_NUMBER, hidden_dim)

        # 2. Multi-Scale Edge Embedding (Dual-Scale RBF)
        self.edge_encoder = DualScaleRBF(
            fine_bins=Config.RBF_FINE_BINS,
            fine_sigma=Config.RBF_FINE_SIGMA,
            coarse_bins=Config.RBF_COARSE_BINS,
            coarse_sigma=Config.RBF_COARSE_SIGMA,
            start=Config.RBF_START,
            end=Config.RBF_END,
            hidden_dim=hidden_dim,
        )

        # 3. Interaction Backbone
        self.layers = nn.ModuleList(
            [
                AdaptiveResidualBlock(hidden_dim, dropout)
                for _ in range(Config.NUM_LAYERS)
            ]
        )

        # 4. Decoupled Readout Heads
        # Head for Formation Energy
        self.head_formation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Head for Bandgap Energy
        self.head_bandgap = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, edge_dist, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Initial Node Features
        h = self.atom_embedding(x)

        # Edge Features (Multi-Scale RBF)
        # edge_dist is assumed to be [num_edges] or [num_edges, 1]
        if edge_dist.dim() > 1:
            edge_dist = edge_dist.squeeze()
        e = self.edge_encoder(edge_dist)

        # Message Passing Layers
        for layer in self.layers:
            h = layer(h, edge_index, e)

        # Global Pooling
        h_pool = global_mean_pool(h, batch)

        # Prediction Heads
        out_form = self.head_formation(h_pool)
        out_band = self.head_bandgap(h_pool)

        # Concatenate outputs [batch_size, 2]
        out = torch.cat([out_form, out_band], dim=1)

        return out
