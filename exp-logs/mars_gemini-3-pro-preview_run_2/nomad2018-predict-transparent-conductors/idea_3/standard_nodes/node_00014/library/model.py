import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import softmax
from library.config import Config


class GaussianRBF(nn.Module):
    """
    Expands scalar distances into a vector of Gaussian Radial Basis Functions.
    This provides a continuous, high-resolution representation of interatomic distances.
    """

    def __init__(self, num_bins, cutoff):
        super().__init__()
        self.num_bins = num_bins
        self.cutoff = cutoff
        # Register centers as buffers so they are part of the state dict but not trainable parameters
        # Linearly spaced centers from 0 to cutoff
        centers = torch.linspace(0, cutoff, num_bins)
        self.register_buffer("centers", centers)

        # Width (sigma) determined by the spacing between centers
        if num_bins > 1:
            width = centers[1] - centers[0]
        else:
            width = 1.0
        # gamma = 1 / (2 * sigma^2), approximating sigma with width
        self.gamma = 1.0 / (width**2)

    def forward(self, distances):
        """
        Args:
            distances: Tensor of shape (num_edges,) representing edge lengths.
        Returns:
            Tensor of shape (num_edges, num_bins)
        """
        # Expand dimensions for broadcasting: (E, 1) - (1, Bins) -> (E, Bins)
        diff = distances.unsqueeze(1) - self.centers.unsqueeze(0)
        return torch.exp(-self.gamma * (diff**2))


class CGCNNConv(MessagePassing):
    """
    Crystal Graph Convolutional Neural Network (CGCNN) Layer.
    Uses a gated activation mechanism with continuous filter convolutions.
    Cite solution_lesson_node_00011: Inductive Bias in Small-Data Regimes
    """

    def __init__(self, atom_fea_len, nbr_fea_len):
        super().__init__(aggr="add", node_dim=0)
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        # Linear layer to produce the gate and core signal
        self.linear = nn.Linear(2 * atom_fea_len + nbr_fea_len, 2 * atom_fea_len)
        self.bn = nn.BatchNorm1d(atom_fea_len)

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node features (N, atom_fea_len)
            edge_index: Graph connectivity (2, E)
            edge_attr: Edge RBF features (E, nbr_fea_len)
        """
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.bn(x + out)  # Residual connection + BatchNorm

    def message(self, x_i, x_j, edge_attr):
        """
        Constructs messages: z_ij = [x_i, x_j, edge_attr]
        out = sigmoid(z W_f + b_f) * softplus(z W_s + b_s)
        """
        z = torch.cat([x_i, x_j, edge_attr], dim=1)
        z = self.linear(z)
        gate, core = z.chunk(2, dim=1)
        return torch.sigmoid(gate) * F.softplus(core)


class DBGT(nn.Module):
    """
    Crystal Graph Convolutional Neural Network (CGCNN).
    Renamed class to DBGT to maintain compatibility with existing training script,
    but the architecture is now CGCNN.
    """

    def __init__(self, config=Config):
        super().__init__()

        self.embed_dim = config.EMBEDDING_DIM
        self.rbf_bins = config.RBF_BINS
        self.cutoff = config.CUTOFF_RADIUS

        # Node Embedding
        self.node_embedding = nn.Embedding(config.MAX_ATOMIC_NUMBER + 1, self.embed_dim)

        # Edge RBF Featurizer
        # Cite solution_lesson_node_00002: Encoding Continuous Spatial Distances via RBF
        self.rbf = GaussianRBF(self.rbf_bins, self.cutoff)

        # Stack of CGCNN Layers
        self.convs = nn.ModuleList(
            [CGCNNConv(self.embed_dim, self.rbf_bins) for _ in range(config.N_LAYERS)]
        )

        # Output Head
        self.output_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Softplus(),
            nn.Linear(self.embed_dim, len(config.TARGET_COLS)),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, data):
        # Unpack data
        if isinstance(data, dict):
            x = data["x"]
            edge_index = data["edge_index"]
            edge_dists = data["edge_attr"]
            batch = data["batch"]
        else:
            x = data.x
            edge_index = data.edge_index
            edge_dists = data.edge_attr
            batch = data.batch

        # 1. Node Embedding
        h = self.node_embedding(x)

        # 2. Edge Featurization (RBF)
        edge_rbf = self.rbf(edge_dists)

        # 3. CGCNN Layers
        for conv in self.convs:
            h = conv(h, edge_index, edge_rbf)

        # 4. Global Pooling
        # CGCNN typically uses mean or softplus-sum pooling. Mean is safer for varying sizes.
        h_graph = global_mean_pool(h, batch)

        # 5. Output Prediction
        out = self.output_head(h_graph)

        return out
