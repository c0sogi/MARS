import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv, global_mean_pool
from library.config import Config


class GaussianSmearing(nn.Module):
    """
    Expands distances using a set of Gaussian Radial Basis Functions (RBF).
    """

    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / ((stop - start) / (num_gaussians - 1)) ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        # dist: [num_edges, 1] -> [num_edges, num_gaussians]
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ElementwiseResidualBlock(nn.Module):
    """
    A block wrapping CGConv with an element-wise learnable residual connection.
    Update rule: h' = Softplus(BatchNorm(CGConv(h, e) + epsilon * h))
    Since PyG's CGConv(h, e) returns (h + Message), adding epsilon * h results in
    (Message + (1 + epsilon) * h), effectively scaling the residual connection.
    """

    def __init__(self, channels, edge_dim, dropout=0.0):
        super().__init__()
        # CGConv expects edge_dim to match the dimension of edge_attr passed to it
        self.conv = CGConv(channels, dim=edge_dim, batch_norm=False, bias=True)
        self.bn = nn.BatchNorm1d(channels)
        self.epsilon = nn.Parameter(torch.zeros(channels))
        self.dropout = nn.Dropout(dropout)
        self.act = nn.Softplus()

    def forward(self, x, edge_index, edge_attr):
        # x: [num_nodes, channels]
        # edge_attr: [num_edges, edge_dim]

        # Standard CGConv forward pass: returns x + Message
        out = self.conv(x, edge_index, edge_attr)

        # Add learnable residual scaling: (x + Message) + epsilon * x = Message + (1 + epsilon) * x
        out = out + self.epsilon * x

        out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)
        return out


class LatticeEncoder(nn.Module):
    """
    Encodes the 6 lattice parameters into a latent embedding.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus(),
        )

    def forward(self, lattice_params):
        return self.net(lattice_params)


class LI_CGCNN_ELR(nn.Module):
    """
    Lattice-Informed Crystal Graph Convolutional Network with Element-wise Learnable Residuals.
    """

    def __init__(self, config):
        super().__init__()

        # Hyperparameters
        self.atom_embedding_dim = config.ATOM_EMBEDDING_DIM
        self.rbf_bins = config.RBF_BINS
        self.lattice_input_dim = config.LATTICE_INPUT_DIM
        self.lattice_emb_dim = config.LATTICE_EMBEDDING_DIM

        # 1. Embeddings
        # Atomic number embedding (assuming max Z=103 from data.py map, using 120 for safety)
        self.embedding = nn.Embedding(120, self.atom_embedding_dim)

        # Edge distance expansion and projection
        self.rbf = GaussianSmearing(
            start=config.RBF_MIN, stop=config.RBF_MAX, num_gaussians=self.rbf_bins
        )
        self.edge_proj = nn.Linear(self.rbf_bins, self.atom_embedding_dim)

        # 2. Local Graph Backbone
        self.conv_layers = nn.ModuleList(
            [
                ElementwiseResidualBlock(
                    channels=self.atom_embedding_dim,
                    edge_dim=self.atom_embedding_dim,
                    dropout=config.DROPOUT_RATE,
                )
                for _ in range(config.NUM_CGCONV_LAYERS)
            ]
        )

        # 3. Lattice Context Stream
        self.lattice_encoder = LatticeEncoder(
            input_dim=self.lattice_input_dim,
            hidden_dim=self.lattice_emb_dim * 2,
            output_dim=self.lattice_emb_dim,
        )

        # 4. Readout Heads
        # Input dimension is graph embedding + lattice embedding
        fused_dim = self.atom_embedding_dim + self.lattice_emb_dim

        # Formation Energy Head
        self.head_formation = nn.Sequential(
            nn.Linear(fused_dim, 64), nn.Softplus(), nn.Linear(64, 1)
        )

        # Bandgap Energy Head
        self.head_bandgap = nn.Sequential(
            nn.Linear(fused_dim, 64), nn.Softplus(), nn.Linear(64, 1)
        )

    def forward(self, data):
        # Unpack data
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        lattice_params = data.lattice_params

        # 1. Node and Edge Embedding
        h = self.embedding(x)  # [num_nodes, atom_emb_dim]

        # Edge features: RBF -> Linear
        edge_feat = self.rbf(edge_attr)  # [num_edges, rbf_bins]
        edge_feat = self.edge_proj(edge_feat)  # [num_edges, atom_emb_dim]

        # 2. Graph Convolutions
        for conv in self.conv_layers:
            h = conv(h, edge_index, edge_feat)

        # 3. Global Pooling
        h_graph = global_mean_pool(h, batch)  # [batch_size, atom_emb_dim]

        # 4. Lattice Encoding
        h_lattice = self.lattice_encoder(
            lattice_params
        )  # [batch_size, lattice_emb_dim]

        # 5. Fusion
        h_fused = torch.cat([h_graph, h_lattice], dim=1)  # [batch_size, fused_dim]

        # 6. Prediction
        out_formation = self.head_formation(h_fused)
        out_bandgap = self.head_bandgap(h_fused)

        # Concatenate outputs: [batch_size, 2]
        return torch.cat([out_formation, out_bandgap], dim=1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
