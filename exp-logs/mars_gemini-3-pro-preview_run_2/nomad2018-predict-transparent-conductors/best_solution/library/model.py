import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from library.config import Config


class RBFExpansion(nn.Module):
    """
    Expands scalar distances into a Gaussian Radial Basis Function (RBF) representation.
    """

    def __init__(self, dmin=0, dmax=8, step=0.2, n_rbf=None):
        super().__init__()
        if n_rbf is None:
            self.filter = torch.arange(dmin, dmax + step, step)
        else:
            self.filter = torch.linspace(dmin, dmax, n_rbf)
            step = (dmax - dmin) / (n_rbf - 1)

        self.register_buffer("centers", self.filter)
        self.sigma = step

    def forward(self, dist):
        """
        Args:
            dist (torch.Tensor): Tensor of shape (E,) containing edge distances.
        Returns:
            torch.Tensor: Tensor of shape (E, n_rbf) containing RBF features.
        """
        return torch.exp(-((dist.unsqueeze(1) - self.centers) ** 2) / (self.sigma**2))


class LRCGCNNLayer(MessagePassing):
    """
    Learnable-Residual Crystal Graph Convolutional Layer.

    Implements the update rule:
    h_{l+1} = Softplus( BatchNorm( Message(h_l, e_ij) + (1 + epsilon) * h_l ) )

    Where Message is the standard CGCNN gated convolution.
    """

    def __init__(self, atom_fea_len, edge_fea_len):
        super().__init__(aggr="add")  # Sum aggregation

        # Input dimension for message computation: node_i + node_j + edge_ij
        self.z_dim = 2 * atom_fea_len + edge_fea_len

        # CGCNN Gated Convolution components
        self.fc_g = nn.Linear(self.z_dim, atom_fea_len)  # Gate function (Sigmoid)
        self.fc_s = nn.Linear(self.z_dim, atom_fea_len)  # Source function (Softplus)

        self.bn = nn.BatchNorm1d(atom_fea_len)
        self.softplus = nn.Softplus()

        # Learnable residual parameter epsilon, initialized to 0.0 for stability
        self.epsilon = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x (torch.Tensor): Node features (N, atom_fea_len)
            edge_index (torch.LongTensor): Graph connectivity (2, E)
            edge_attr (torch.Tensor): Edge features (E, edge_fea_len)
        """
        # Compute and aggregate messages
        # propagate calls message() internally
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Apply Learnable Residual Connection
        # The residual weight (1 + epsilon) allows the model to learn to amplify
        # the identity path if preserving atomic identity is beneficial.
        out = out + (1.0 + self.epsilon) * x

        # Batch Normalization and Activation
        out = self.bn(out)
        out = self.softplus(out)

        return out

    def message(self, x_i, x_j, edge_attr):
        """
        CGCNN message function: z_ij W_f * sigma(z_ij W_s)
        """
        # Concatenate source node, target node, and edge features
        z = torch.cat([x_i, x_j, edge_attr], dim=1)

        # Gated convolution
        gate = torch.sigmoid(self.fc_g(z))
        source = self.softplus(self.fc_s(z))

        return gate * source


class LRCGCNN(nn.Module):
    """
    Learnable-Residual Crystal Graph Neural Network (LR-CGCNN) with Dual Pooling.
    """

    def __init__(
        self,
        atom_fea_len=Config.ATOM_FEA_LEN,
        h_fea_len=Config.H_FEA_LEN,
        n_conv=Config.N_CONV,
        n_h=Config.N_H,
        n_rbf=Config.N_RBF,
        radius=Config.RADIUS,
    ):
        super().__init__()

        # 1. Initial Embeddings
        # Embedding for atomic numbers (1-100)
        self.embedding = nn.Embedding(100, atom_fea_len)

        # Edge distance expansion and projection
        self.rbf_expansion = RBFExpansion(dmin=0, dmax=radius, n_rbf=n_rbf)
        self.edge_embedding = nn.Linear(n_rbf, atom_fea_len)

        # 2. Interaction Layers
        self.convs = nn.ModuleList(
            [
                LRCGCNNLayer(atom_fea_len=atom_fea_len, edge_fea_len=atom_fea_len)
                for _ in range(n_conv)
            ]
        )

        # 3. Prediction Heads
        # Input is atom_fea_len because of Mean Pooling (Cite Lesson 33)
        pool_dim = atom_fea_len

        self.fc_form = self._build_mlp(pool_dim, h_fea_len, 1, n_h)
        self.fc_band = self._build_mlp(pool_dim, h_fea_len, 1, n_h)

        self.dropout = nn.Dropout(Config.DROPOUT)

    def _build_mlp(self, input_dim, hidden_dim, output_dim, n_layers):
        """
        Helper to build a Multi-Layer Perceptron.
        """
        layers = []
        dims = [input_dim] + [hidden_dim] * n_layers + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                # Activation and Dropout for hidden layers
                layers.append(nn.Softplus())
                layers.append(nn.Dropout(Config.DROPOUT))

        return nn.Sequential(*layers)

    def forward(self, data):
        """
        Forward pass of the model.
        """
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Node Embedding
        x = self.embedding(x)  # (N, F)

        # Edge Embedding
        edge_attr = self.rbf_expansion(edge_attr)  # (E, n_rbf)
        edge_attr = self.edge_embedding(edge_attr)  # (E, F)

        # Apply GNN Layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = self.dropout(x)

        # Global Mean Pooling (Cite Lesson 33)
        # Prefer parameter-free mean pooling for intensive physical properties
        x_pool = global_mean_pool(x, batch)  # (B, F)

        # Decoupled Prediction Heads
        out_form = self.fc_form(x_pool)
        out_band = self.fc_band(x_pool)

        # Concatenate outputs for final prediction (B, 2)
        out = torch.cat([out_form, out_band], dim=1)

        return out
