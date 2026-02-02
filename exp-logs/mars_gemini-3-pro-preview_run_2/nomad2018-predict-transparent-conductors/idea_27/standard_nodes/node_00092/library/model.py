import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from library.config import Config


class GaussianSmearing(nn.Module):
    """
    Expands distances into a vector of Gaussian radial basis functions.
    """

    def __init__(self, start=0.0, stop=5.0, n_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, n_gaussians)
        # The width (coeff) is determined by the spacing between gaussians
        self.coeff = -0.5 / ((stop - start) / (n_gaussians - 1)) ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        # dist: [num_edges] -> [num_edges, 1]
        # offset: [n_gaussians] -> [1, n_gaussians]
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ReceiverAwareConv(MessagePassing):
    """
    Receiver-Aware Graph Convolution.
    Constructs messages using Source, Target (Receiver), and Edge features.
    """

    def __init__(self, node_dim, edge_dim):
        super().__init__(aggr="add")  # Sum aggregation
        self.emb_dim = node_dim
        self.edge_dim = edge_dim

        # Input dim: Target Node + Source Node + Edge Feature
        in_dim = 2 * node_dim + edge_dim

        self.lin1 = nn.Linear(in_dim, node_dim)
        self.lin2 = nn.Linear(in_dim, node_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.zeros_(self.lin1.bias)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, x, edge_index, edge_attr):
        # x: [num_nodes, node_dim]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, edge_dim]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i: target nodes (receiver)
        # x_j: source nodes
        # Concatenate all available local information
        z_ij = torch.cat([x_i, x_j, edge_attr], dim=-1)

        # Gated activation: Content * Gate
        # Softplus is used for content to maintain positive/smooth signal characteristics
        content = F.softplus(self.lin1(z_ij))
        gate = torch.sigmoid(self.lin2(z_ij))

        return content * gate


class InteractionBlock(nn.Module):
    """
    Encapsulates Convolution, Normalization, and Learnable Residual.
    """

    def __init__(self, hidden_dim, edge_dim, dropout_rate=0.0, residual_init=0.0):
        super().__init__()
        self.conv = ReceiverAwareConv(hidden_dim, edge_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # Learnable scalar for residual connection
        # Allows the model to learn how much of the identity to preserve
        self.epsilon = nn.Parameter(torch.tensor(residual_init))

    def forward(self, x, edge_index, edge_attr):
        # 1. Message Passing
        m = self.conv(x, edge_index, edge_attr)

        # 2. Residual Connection with Learnable Scalar
        # Apply residual before normalization to prevent signal washout (Cite Lesson 90)
        m = m + (1.0 + self.epsilon) * x

        # 3. Batch Normalization on the sum
        m = self.bn(m)

        # 4. Activation
        out = F.softplus(m)

        out = self.dropout(out)
        return out


class SPRACGN(nn.Module):
    """
    Stoichiometry-Preserving Receiver-Aware Crystal Graph Network.
    """

    def __init__(self, config=Config):
        super().__init__()

        # Hyperparameters
        self.atom_embedding_dim = config.ATOM_EMBEDDING_DIM
        self.hidden_dim = config.HIDDEN_DIM
        self.edge_embedding_dim = config.EDGE_EMBEDDING_DIM
        self.rbf_bins = config.RBF_BINS
        self.rbf_lower = config.RBF_LOWER
        self.rbf_upper = config.RBF_UPPER
        self.num_blocks = config.NUM_INTERACTION_BLOCKS
        self.dropout_rate = config.DROPOUT_RATE
        self.residual_init = config.RESIDUAL_INIT

        # 1. Embeddings
        # Atomic numbers up to ~100
        self.embedding = nn.Embedding(100, self.atom_embedding_dim)

        # Edge Features
        self.distance_expansion = GaussianSmearing(
            start=self.rbf_lower, stop=self.rbf_upper, n_gaussians=self.rbf_bins
        )
        # Shared projection for edge features
        self.edge_projection = nn.Linear(self.rbf_bins, self.edge_embedding_dim)

        # 2. Interaction Backbone
        self.blocks = nn.ModuleList(
            [
                InteractionBlock(
                    hidden_dim=self.hidden_dim,
                    edge_dim=self.edge_embedding_dim,
                    dropout_rate=self.dropout_rate,
                    residual_init=self.residual_init,
                )
                for _ in range(self.num_blocks)
            ]
        )

        # 3. Stoichiometric Skip Readout
        # The readout fuses learned structural features with initial compositional features
        combined_dim = self.hidden_dim + self.atom_embedding_dim

        # Prediction Head: Formation Energy
        self.head_energy = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 1),
        )

        # Prediction Head: Bandgap Energy
        self.head_bandgap = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        z, edge_index, edge_attr, batch = (
            data.z,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Initial Node Embeddings (h^0)
        h0 = self.embedding(z)

        # Edge Processing
        edge_features = self.distance_expansion(edge_attr)
        edge_features = self.edge_projection(edge_features)

        # Structural Evolution
        h = h0
        for block in self.blocks:
            h = block(h, edge_index, edge_features)

        # h now contains h^L (structurally contextualized embeddings)

        # Dual Pooling
        # 1. Structural Feature: Mean pool of final embeddings
        z_struct = global_mean_pool(h, batch)

        # 2. Compositional Feature: Mean pool of initial embeddings
        # This preserves the exact stoichiometry (e.g., "50% Al, 50% Ga") without
        # the smoothing effects of message passing.
        z_comp = global_mean_pool(h0, batch)

        # Late Fusion
        z_final = torch.cat([z_struct, z_comp], dim=1)

        # Predictions
        out_energy = self.head_energy(z_final)
        out_bandgap = self.head_bandgap(z_final)

        return torch.cat([out_energy, out_bandgap], dim=1)
