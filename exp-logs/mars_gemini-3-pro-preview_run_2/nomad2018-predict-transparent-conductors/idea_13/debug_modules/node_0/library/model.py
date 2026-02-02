import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from library.config import Config


class GaussianRBF(nn.Module):
    """
    Expands scalar distances into a vector of Gaussian Radial Basis Functions.
    """

    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        # Calculate coefficient such that gaussians overlap reasonably
        # Width is related to the spacing between centers
        self.coeff = -0.5 / ((stop - start) / (num_gaussians - 1)) ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        # dist: [num_edges, 1]
        # offset: [num_gaussians]
        # Result: [num_edges, num_gaussians]
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class EdgeEncoder(nn.Module):
    """
    Non-linear encoding of RBF expanded edge features using an MLP.
    Linear -> Softplus -> Linear
    """

    def __init__(self, num_rbf, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, edge_attr):
        return self.mlp(edge_attr)


class CGCNNConv(MessagePassing):
    """
    Gated convolution layer inspired by CGCNN.
    """

    def __init__(self, node_dim, edge_dim):
        super().__init__(aggr="add")
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        # Maps concatenation of (x_i, x_j, e_ij) to (filter, gate)
        self.lin = nn.Linear(2 * node_dim + edge_dim, 2 * node_dim)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # Concatenate source node, target node, and edge features
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        z = self.lin(z)

        # Split into filter and gate parts
        # Standard CGCNN: sigmoid(z * W_f + b_f) * softplus(z * W_s + b_s)
        # Here we use Softplus for filter and Sigmoid for gate as per common implementations
        a, b = z.chunk(2, dim=-1)
        return F.softplus(a) * torch.sigmoid(b)


class PreActGatedConvBlock(nn.Module):
    """
    Interaction block with Pre-Activation Residual connection.
    Structure: Input -> LayerNorm -> GatedConv -> Dropout -> + Residual
    """

    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.conv = CGCNNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        residual = x

        # Pre-activation
        x = self.norm(x)

        # Convolution
        x = self.conv(x, edge_index, edge_attr)

        # Regularization
        x = self.dropout(x)

        # Residual connection
        return residual + x


class CrystalGraphResNet(nn.Module):
    """
    Main model architecture:
    1. Node Embedding
    2. Edge RBF + Non-linear Encoding
    3. Stack of Pre-Activated Gated Convolution Blocks
    4. Global Mean Pooling
    5. Decoupled Prediction Heads
    """

    def __init__(self, config=Config):
        super().__init__()

        # Node Embedding
        # Embed atomic numbers (Z) into dense vectors
        # Assuming atomic numbers < 100
        self.embedding = nn.Embedding(100, config.ATOM_EMBEDDING_DIM)

        # Edge Encoding
        self.rbf = GaussianRBF(
            start=config.RBF_MIN, stop=config.RBF_MAX, num_gaussians=config.NUM_RBF_BINS
        )
        self.edge_encoder = EdgeEncoder(config.NUM_RBF_BINS, config.HIDDEN_DIM)

        # Backbone: Stack of Interaction Blocks
        self.blocks = nn.ModuleList(
            [
                PreActGatedConvBlock(config.HIDDEN_DIM, config.DROPOUT)
                for _ in range(config.NUM_LAYERS)
            ]
        )

        # Prediction Heads
        # Decoupled MLPs for each target

        # Head for formation energy
        self.head_form = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(config.HIDDEN_DIM, 1),
        )

        # Head for bandgap energy
        self.head_gap = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(config.HIDDEN_DIM, 1),
        )

    def forward(self, data):
        # Unpack data
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # 1. Embed Nodes
        x = self.embedding(x)

        # 2. Encode Edges
        edge_feat = self.rbf(edge_attr)
        edge_feat = self.edge_encoder(edge_feat)

        # 3. Message Passing Backbone
        for block in self.blocks:
            x = block(x, edge_index, edge_feat)

        # 4. Global Pooling
        x = global_mean_pool(x, batch)

        # 5. Prediction Heads
        out_form = self.head_form(x)
        out_gap = self.head_gap(x)

        # Concatenate predictions
        return torch.cat([out_form, out_gap], dim=1)
