import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from library.config import Config


class GaussianSmearing(nn.Module):
    """
    Expands distances using a static Gaussian Radial Basis Function (RBF) filter.
    """

    def __init__(self, start=0.0, stop=5.0, num_gaussians=32, sigma=0.25):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (sigma**2)
        self.register_buffer("offset", offset)

    def forward(self, dist):
        # dist: (num_edges, )
        # offset: (num_gaussians, )
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ReceiverAwareConv(MessagePassing):
    """
    Receiver-Aware Gated Convolution.
    Concatenates Target Node, Source Node, and Edge Feature.
    Applies a gated mechanism with Softplus content and Sigmoid gate.
    """

    def __init__(self, node_dim, edge_dim):
        super().__init__(aggr="add")  # Aggregation is sum

        # Input dimension: h_i + h_j + e_ij
        input_dim = 2 * node_dim + edge_dim

        self.linear_content = nn.Linear(input_dim, node_dim)
        self.linear_gate = nn.Linear(input_dim, node_dim)

        self.activation_content = nn.Softplus()
        self.activation_gate = nn.Sigmoid()

    def forward(self, x, edge_index, edge_attr):
        # x: (num_nodes, node_dim)
        # edge_index: (2, num_edges)
        # edge_attr: (num_edges, edge_dim)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i: Target node features (num_edges, node_dim)
        # x_j: Source node features (num_edges, node_dim)
        # edge_attr: Edge features (num_edges, edge_dim)

        # Concatenate [h_i || h_j || e_ij]
        z_ij = torch.cat([x_i, x_j, edge_attr], dim=-1)

        # Compute message: Softplus(Linear1(z)) * Sigmoid(Linear2(z))
        content = self.activation_content(self.linear_content(z_ij))
        gate = self.activation_gate(self.linear_gate(z_ij))

        return content * gate


class AdaptiveResidualBlock(nn.Module):
    """
    Stabilized Adaptive Residual Block.
    Applies Batch Normalization to the sum of convolution output and adaptive residual.
    """

    def __init__(self, node_dim, dropout_rate):
        super().__init__()
        # Learnable scalar epsilon initialized to 0
        self.epsilon = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm1d(node_dim)
        self.activation = nn.Softplus()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, h_in, m_out):
        # h_in: Input node features from previous layer (h_l)
        # m_out: Aggregated messages from convolution (Agg(m_ij))

        # h_{l+1} = Softplus(BatchNorm(Agg(m_ij) + (1 + epsilon) * h_l))
        residual = (1 + self.epsilon) * h_in
        out = m_out + residual

        out = self.bn(out)
        out = self.activation(out)
        out = self.dropout(out)

        return out


class SRACGN(nn.Module):
    """
    Smoothed Receiver-Aware Crystal Graph Network.
    """

    def __init__(self, config=Config):
        super().__init__()

        self.node_dim = config.ATOM_EMBEDDING_DIM
        self.edge_dim = config.ATOM_EMBEDDING_DIM  # Projected edge dim matches node dim

        # 1. Embeddings & Smoothed Edge Encoding
        # Atomic numbers up to ~100. Using 118 to be safe.
        self.embedding = nn.Embedding(118, self.node_dim)

        self.rbf = GaussianSmearing(
            start=0.0,
            stop=config.CUTOFF,
            num_gaussians=config.RBF_BINS,
            sigma=config.RBF_SIGMA,
        )

        # Shared Linear Layer for edge projection
        self.edge_proj = nn.Linear(config.RBF_BINS, self.edge_dim)

        # 2. & 3. Interaction Backbone (Conv + Residual)
        self.convs = nn.ModuleList()
        self.res_blocks = nn.ModuleList()

        for _ in range(config.NUM_INTERACTION_BLOCKS):
            self.convs.append(ReceiverAwareConv(self.node_dim, self.edge_dim))
            self.res_blocks.append(
                AdaptiveResidualBlock(self.node_dim, config.DROPOUT_RATE)
            )

        # 4. Readout Heads
        # Decoupled MLPs for formation energy and bandgap
        # Using a hidden layer of size node_dim // 2
        hidden_dim = self.node_dim // 2

        self.head_formation = nn.Sequential(
            nn.Linear(self.node_dim, hidden_dim),
            nn.Softplus(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(hidden_dim, 1),
        )

        self.head_bandgap = nn.Sequential(
            nn.Linear(self.node_dim, hidden_dim),
            nn.Softplus(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Initial Node Embedding
        h = self.embedding(x)

        # Edge Encoding and Projection
        # edge_attr comes in as distances (num_edges, 1) -> flatten to (num_edges,)
        edge_dist = edge_attr.squeeze(-1)
        rbf_features = self.rbf(edge_dist)
        e_proj = self.edge_proj(rbf_features)

        # Interaction Blocks
        for conv, res_block in zip(self.convs, self.res_blocks):
            # Convolution (Message Passing)
            m = conv(h, edge_index, e_proj)

            # Residual Update
            h = res_block(h, m)

        # Global Pooling
        h_pool = global_mean_pool(h, batch)

        # Prediction Heads
        out_formation = self.head_formation(h_pool)
        out_bandgap = self.head_bandgap(h_pool)

        # Concatenate outputs: (batch_size, 2)
        return torch.cat([out_formation, out_bandgap], dim=1)
