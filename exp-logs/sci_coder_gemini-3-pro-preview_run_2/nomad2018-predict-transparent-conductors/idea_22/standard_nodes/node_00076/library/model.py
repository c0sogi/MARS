import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from library.config import Config


class GaussianRBF(nn.Module):
    """
    Expands scalar distances into a Gaussian Radial Basis Function (RBF) representation.
    """

    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        # The coefficient is set such that the gaussians overlap effectively
        self.coeff = -0.5 / ((stop - start) / (num_gaussians - 1)) ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        # dist: [num_edges] -> [num_edges, 1]
        # offset: [num_gaussians] -> [1, num_gaussians]
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ReceiverAwareConv(MessagePassing):
    """
    Receiver-Aware Gated Convolution layer.
    Constructs messages by explicitly concatenating source node, target node,
    and projected edge features to capture pairwise chemistry.
    """

    def __init__(self, node_dim, edge_dim):
        super().__init__(aggr="add")  # Aggregation method: sum
        self.emb_dim = node_dim
        self.edge_dim = edge_dim

        # Project RBF edge features to embedding dimension
        self.edge_lin = nn.Linear(Config.RBF_BINS, edge_dim)

        # Input dimension for filter/core generation:
        # Target Node (x_i) + Source Node (x_j) + Projected Edge (e'_ij)
        input_dim = 2 * node_dim + edge_dim

        # Filter and Core generators
        self.lin_filter = nn.Linear(input_dim, node_dim)
        self.lin_core = nn.Linear(input_dim, node_dim)

        self.bn_filter = nn.BatchNorm1d(node_dim)
        self.bn_core = nn.BatchNorm1d(node_dim)

    def forward(self, x, edge_index, edge_attr):
        # Project edge features first
        e_emb = self.edge_lin(edge_attr)

        # Propagate messages
        return self.propagate(edge_index, x=x, edge_attr=e_emb)

    def message(self, x_i, x_j, edge_attr):
        """
        x_i: Features of target nodes (receivers)
        x_j: Features of source nodes (senders)
        edge_attr: Projected edge features
        """
        # Concatenate receiver, sender, and edge info
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)

        # Compute Gated Convolution components
        # Filter: Sigmoid activation
        filter_out = self.lin_filter(z)
        filter_out = self.bn_filter(filter_out)
        filter_out = torch.sigmoid(filter_out)

        # Core: Softplus activation
        core_out = self.lin_core(z)
        core_out = self.bn_core(core_out)
        core_out = F.softplus(core_out)

        # Element-wise product
        return filter_out * core_out


class InteractionBlock(nn.Module):
    """
    Interaction Block with Adaptive Residual Connections.
    Applies the convolution followed by a learnable residual update.
    """

    def __init__(self, node_dim, edge_dim, dropout_rate=0.0):
        super().__init__()
        self.conv = ReceiverAwareConv(node_dim, edge_dim)
        self.bn = nn.BatchNorm1d(node_dim)

        # Learnable scalar alpha initialized to 0
        self.alpha = nn.Parameter(torch.zeros(1))

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, edge_attr):
        # Convolution step
        h_conv = self.conv(x, edge_index, edge_attr)

        # Adaptive Residual: Softplus(BN(Conv(h) + (1 + alpha) * h))
        # We scale the identity path by (1 + alpha)
        h_residual = (1 + self.alpha) * x

        # Sum before BatchNorm
        h_sum = h_conv + h_residual

        # BatchNorm and Activation
        h_bn = self.bn(h_sum)
        h_out = F.softplus(h_bn)

        h_out = self.dropout(h_out)
        return h_out


class RA_CGN_AR(nn.Module):
    """
    Receiver-Aware Crystal Graph Network with Adaptive Residuals.
    """

    def __init__(self):
        super().__init__()

        # Hyperparameters from Config
        self.atom_dim = Config.ATOM_EMBEDDING_DIM
        self.edge_dim = Config.EDGE_EMBEDDING_DIM
        self.num_layers = Config.NUM_LAYERS
        self.dropout_rate = Config.DROPOUT_RATE

        # 1. Node Embedding: Atomic numbers (1-100) -> Embedding Dim
        self.embedding = nn.Embedding(100, self.atom_dim)

        # 2. Edge Expansion: Gaussian RBF
        self.rbf = GaussianRBF(
            start=Config.RBF_LOWER, stop=Config.RBF_UPPER, num_gaussians=Config.RBF_BINS
        )

        # 3. Interaction Blocks
        self.blocks = nn.ModuleList(
            [
                InteractionBlock(self.atom_dim, self.edge_dim, self.dropout_rate)
                for _ in range(self.num_layers)
            ]
        )

        # 4. Readout Heads
        # Head for Formation Energy
        self.head_formation = nn.Sequential(
            nn.Linear(self.atom_dim, self.atom_dim),
            nn.Softplus(),
            nn.Linear(self.atom_dim, 1),
        )

        # Head for Bandgap Energy
        self.head_bandgap = nn.Sequential(
            nn.Linear(self.atom_dim, self.atom_dim),
            nn.Softplus(),
            nn.Linear(self.atom_dim, 1),
        )

    def forward(self, data):
        """
        Forward pass of the model.
        Expects a PyG Data object with x (atomic numbers), edge_index, edge_attr (distances), and batch.
        """
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Initial Embedding of atoms
        h = self.embedding(x)

        # Expand edge distances using RBF
        edge_rbf = self.rbf(edge_attr)

        # Pass through Interaction Blocks
        for block in self.blocks:
            h = block(h, edge_index, edge_rbf)

        # Global Mean Pooling
        h_pool = global_mean_pool(h, batch)

        # Predict targets using separate heads
        out_formation = self.head_formation(h_pool)
        out_bandgap = self.head_bandgap(h_pool)

        # Concatenate outputs: [batch_size, 2]
        return torch.cat([out_formation, out_bandgap], dim=1)
