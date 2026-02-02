import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from library.config import Config


class RBFExpansion(nn.Module):
    """
    Expands scalar distances into a vector of Radial Basis Functions (Gaussian).
    """

    def __init__(self, vmin=0.0, vmax=5.0, bins=60, lengthscale=None):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        # Create centers for the Gaussian functions
        self.centers = torch.linspace(vmin, vmax, bins)

        if lengthscale is None:
            # Default lengthscale to the step size
            self.lengthscale = (vmax - vmin) / bins
        else:
            self.lengthscale = lengthscale

        self.gamma = 1.0 / (self.lengthscale**2)

        # Register as buffer to be part of the model state but not a learnable parameter
        self.register_buffer("centers_tensor", self.centers)

    def forward(self, distance):
        """
        Args:
            distance (torch.Tensor): Tensor of shape [E, 1] containing edge distances.
        Returns:
            torch.Tensor: Tensor of shape [E, bins] containing RBF features.
        """
        # distance: [E, 1] -> [E, 1] - [bins] (broadcast) -> [E, bins]
        return torch.exp(-self.gamma * (distance - self.centers_tensor) ** 2)


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Layer.
    Conditions node features on global crystal properties.
    """

    def __init__(self, global_dim, channels):
        super().__init__()
        self.channels = channels
        # Project global features to generate scale (gamma) and shift (beta)
        self.proj = nn.Sequential(
            nn.Linear(global_dim, channels),
            nn.SiLU(),
            nn.Linear(channels, 2 * channels),
        )

    def forward(self, x, global_feat, batch):
        """
        Args:
            x (torch.Tensor): Node features [N, C].
            global_feat (torch.Tensor): Global features [B, G].
            batch (torch.Tensor): Batch indices for nodes [N].
        Returns:
            torch.Tensor: Modulated node features [N, C].
        """
        # Map global features to each node based on batch index
        node_global = global_feat[batch]  # [N, G]

        # Generate parameters
        params = self.proj(node_global)  # [N, 2*C]
        gamma, beta = torch.split(params, self.channels, dim=1)  # [N, C], [N, C]

        # Apply affine transformation: x' = (1 + gamma) * x + beta
        return x * (1.0 + gamma) + beta


class CGCNNConv(MessagePassing):
    """
    Crystal Graph Convolutional Neural Network (CGCNN) Layer.
    """

    def __init__(self, node_dim, edge_dim):
        super().__init__(aggr="add")  # Sum aggregation
        self.emb_dim = node_dim
        self.edge_dim = edge_dim

        # Input dimension for the linear layers: 2 * node_features + edge_features
        # Corresponds to concatenation of source node, target node, and edge attributes
        in_dim = 2 * node_dim + edge_dim

        self.lin_f = nn.Linear(in_dim, node_dim)  # Filter function
        self.lin_s = nn.Linear(in_dim, node_dim)  # Core/Sigmoid function

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x (torch.Tensor): Node features [N, node_dim].
            edge_index (torch.Tensor): Graph connectivity [2, E].
            edge_attr (torch.Tensor): Edge features [E, edge_dim].
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        """
        Constructs messages: sigmoid(Filter) * softplus(Core)
        """
        # Concatenate source node, target node, and edge features
        z = torch.cat([x_i, x_j, edge_attr], dim=1)

        # Calculate gates and core features
        gate = torch.sigmoid(self.lin_f(z))
        core = F.softplus(self.lin_s(z))

        return gate * core


class GCCGCNN(nn.Module):
    """
    Global-Conditioned Crystal Graph Convolutional Neural Network.
    """

    def __init__(self, config=Config):
        super().__init__()

        # Hyperparameters
        atom_dim = config.ATOM_EMBEDDING_DIM
        edge_dim = config.EDGE_EMBEDDING_DIM
        global_dim = config.GLOBAL_INPUT_DIM

        # 1. Embeddings
        # Atomic number embedding
        self.atom_embedding = nn.Embedding(config.MAX_ATOMIC_NUMBER + 1, atom_dim)

        # Edge distance expansion and embedding
        self.rbf = RBFExpansion(
            vmin=config.RBF_MIN, vmax=config.RBF_MAX, bins=config.RBF_BINS
        )
        self.edge_embedding = nn.Linear(config.RBF_BINS, edge_dim)

        # Global feature encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, atom_dim), nn.SiLU(), nn.Linear(atom_dim, atom_dim)
        )

        # 2. Interaction Blocks
        self.blocks = nn.ModuleList()
        for _ in range(config.NUM_FILM_BLOCKS):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "film": FiLMLayer(atom_dim, atom_dim),
                        "conv": CGCNNConv(atom_dim, edge_dim),
                        "bn": nn.BatchNorm1d(atom_dim),
                    }
                )
            )

        # 3. Readout and Prediction
        # We concatenate the pooled graph features with the global features
        concat_dim = atom_dim + atom_dim

        self.final_mlp = nn.Sequential(
            nn.Linear(concat_dim, config.MLP_HIDDEN_DIM),
            nn.SiLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.MLP_HIDDEN_DIM, len(config.TARGET_COLS)),
        )

    def forward(self, data):
        # Unpack PyG Data object
        x, edge_index, edge_attr, global_feat, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.global_feat,
            data.batch,
        )

        # Initial Node Embeddings
        h = self.atom_embedding(x)  # [N, atom_dim]

        # Edge Embeddings
        e_rbf = self.rbf(edge_attr)  # [E, bins]
        e = self.edge_embedding(e_rbf)  # [E, edge_dim]

        # Global Embeddings
        g = self.global_encoder(global_feat)  # [B, atom_dim]

        # Interaction Blocks
        for block in self.blocks:
            # 1. Global Conditioning via FiLM
            # Modulates node features based on the crystal's global properties
            h_film = block["film"](h, g, batch)

            # 2. Local Message Passing via CGCNN
            h_conv = block["conv"](h_film, edge_index, e)

            # 3. Residual Connection
            h = h + h_conv

            # 4. Normalization and Activation
            h = block["bn"](h)
            h = F.silu(h)

        # Global Pooling (Mean)
        h_pool = global_mean_pool(h, batch)  # [B, atom_dim]

        # Concatenate Graph Features with Global Features
        out = torch.cat([h_pool, g], dim=1)  # [B, 2*atom_dim]

        # Final Prediction
        pred = self.final_mlp(out)  # [B, 2]

        return pred
