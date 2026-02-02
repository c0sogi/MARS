import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max
from library.config import Config


class MLPBlock(nn.Module):
    """
    A standard building block for MLPs consisting of:
    Linear -> Batch Normalization -> ReLU -> Dropout.
    """

    def __init__(self, input_dim, output_dim, dropout_rate, use_batch_norm=True):
        super(MLPBlock, self).__init__()
        layers = [nn.Linear(input_dim, output_dim)]
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(output_dim))
        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class AtomicEncoder(nn.Module):
    """
    Processes atomic-level features (One-hot, Coords, NN Dist, Anisotropy Eigenvalues).
    Uses a Wide MLP structure to project into a high-dimensional latent space.
    """

    def __init__(self):
        super(AtomicEncoder, self).__init__()
        input_dim = Config.ATOMIC_INPUT_DIM
        hidden_dim = Config.ATOMIC_HIDDEN_DIM
        num_layers = Config.ATOMIC_LAYERS
        dropout = Config.DROPOUT_RATE
        use_bn = Config.USE_BATCH_NORM

        layers = []
        # Input layer
        layers.append(MLPBlock(input_dim, hidden_dim, dropout, use_bn))

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(MLPBlock(hidden_dim, hidden_dim, dropout, use_bn))

        # Output projection (Linear only, no activation for the final embedding before pooling)
        layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class GlobalEncoder(nn.Module):
    """
    Processes global crystal features (Lattice, Volume, Density, Stoichiometry).
    """

    def __init__(self):
        super(GlobalEncoder, self).__init__()
        input_dim = Config.GLOBAL_INPUT_DIM
        hidden_dim = Config.GLOBAL_HIDDEN_DIM
        num_layers = Config.GLOBAL_LAYERS
        dropout = Config.DROPOUT_RATE
        use_bn = Config.USE_BATCH_NORM

        layers = []
        # Input layer
        layers.append(MLPBlock(input_dim, hidden_dim, dropout, use_bn))

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(MLPBlock(hidden_dim, hidden_dim, dropout, use_bn))

        # Output projection
        layers.append(MLPBlock(hidden_dim, hidden_dim, dropout, use_bn))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class FusionHead(nn.Module):
    """
    Merges aggregated atomic embeddings and global embeddings to predict targets.
    """

    def __init__(self):
        super(FusionHead, self).__init__()
        # Input: Mean Pooled Atomic (H_a) + Max Pooled Atomic (H_a) + Global (H_g)
        input_dim = (2 * Config.ATOMIC_HIDDEN_DIM) + Config.GLOBAL_HIDDEN_DIM
        hidden_dim = Config.FUSION_HIDDEN_DIM
        output_dim = Config.OUTPUT_DIM
        num_layers = Config.FUSION_LAYERS
        dropout = Config.DROPOUT_RATE
        use_bn = Config.USE_BATCH_NORM

        layers = []
        # Input layer
        layers.append(MLPBlock(input_dim, hidden_dim, dropout, use_bn))

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(MLPBlock(hidden_dim, hidden_dim, dropout, use_bn))

        # Final regression layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LAWDS(nn.Module):
    """
    Local-Anisotropy Enhanced Wide Deep Sets (LA-WDS).

    Architecture:
    1. Atomic Stream: Processes individual atoms with local anisotropy features.
    2. Aggregation: Dual pooling (Mean + Max) of atomic embeddings.
    3. Global Stream: Processes macroscopic crystal properties.
    4. Fusion: Concatenates aggregated atomic and global features.
    5. Head: Regresses formation energy and bandgap.
    """

    def __init__(self):
        super(LAWDS, self).__init__()
        self.atomic_encoder = AtomicEncoder()
        self.global_encoder = GlobalEncoder()
        self.fusion_head = FusionHead()

    def forward(self, atom_x, batch_indices, global_x):
        """
        Args:
            atom_x (Tensor): [Total_Atoms, ATOMIC_INPUT_DIM]
            batch_indices (Tensor): [Total_Atoms] mapping atoms to batch index
            global_x (Tensor): [Batch_Size, GLOBAL_INPUT_DIM]

        Returns:
            Tensor: [Batch_Size, 2] predictions
        """
        # 1. Atomic Stream
        # [Total_Atoms, ATOMIC_HIDDEN_DIM]
        atom_emb = self.atomic_encoder(atom_x)

        # 2. Aggregation (Dual Pooling)
        # Scatter mean: [Batch_Size, ATOMIC_HIDDEN_DIM]
        mean_pool = scatter_mean(atom_emb, batch_indices, dim=0)
        # Scatter max: [Batch_Size, ATOMIC_HIDDEN_DIM]
        max_pool, _ = scatter_max(atom_emb, batch_indices, dim=0)

        # 3. Global Stream
        # [Batch_Size, GLOBAL_HIDDEN_DIM]
        global_emb = self.global_encoder(global_x)

        # 4. Late Fusion
        # Concatenate: [Batch_Size, 2*ATOMIC_HIDDEN + GLOBAL_HIDDEN]
        fused_features = torch.cat([mean_pool, max_pool, global_emb], dim=1)

        # 5. Prediction
        # [Batch_Size, 2]
        output = self.fusion_head(fused_features)

        return output
