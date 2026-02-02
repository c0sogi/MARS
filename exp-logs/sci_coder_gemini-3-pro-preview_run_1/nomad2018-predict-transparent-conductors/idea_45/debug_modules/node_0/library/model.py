import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_scatter import scatter_mean, scatter_max
except ImportError:
    # Fallback for environments where torch_scatter might have issues,
    # though it is listed as installed.
    pass

from library.config import Config


class AtomicStream(nn.Module):
    """
    Chemically-Explicit Point Processor.
    Projects atomic features (Identity, NN, Coords, Contexts) into a high-dimensional latent space.
    """

    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class GlobalStream(nn.Module):
    """
    Variance-Aware Context Processor.
    Projects global crystal features (Lattice, Stoichiometry, Physical Variance) into latent space.
    """

    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CEAMSDS(nn.Module):
    """
    Chemically-Explicit Anisotropic Multi-Scale Deep Sets.

    Architecture:
    1. Atomic Stream -> Dense Embedding per atom
    2. Global Stream -> Dense Embedding per crystal
    3. Dual Pooling (Mean + Max) of Atomic Embeddings via Scatter operations
    4. Late Fusion of Pooled Atomic + Global Embeddings
    5. Regression Head -> Targets
    """

    def __init__(self):
        super().__init__()

        # 1. Atomic Stream
        self.atomic_stream = AtomicStream(
            input_dim=Config.ATOMIC_INPUT_DIM,
            hidden_dim=Config.ATOMIC_HIDDEN_DIM,
            dropout=Config.DROPOUT,
        )

        # 2. Global Stream
        self.global_stream = GlobalStream(
            input_dim=Config.GLOBAL_INPUT_DIM,
            hidden_dim=Config.GLOBAL_HIDDEN_DIM,
            dropout=Config.DROPOUT,
        )

        # 3. Fusion Head
        # Concatenation of:
        #   - Atomic Mean Pooling (ATOMIC_HIDDEN_DIM)
        #   - Atomic Max Pooling  (ATOMIC_HIDDEN_DIM)
        #   - Global Embedding    (GLOBAL_HIDDEN_DIM)
        fusion_input_dim = (Config.ATOMIC_HIDDEN_DIM * 2) + Config.GLOBAL_HIDDEN_DIM

        layers = []
        in_dim = fusion_input_dim

        # Build dynamic MLP head based on config
        for h_dim in Config.FUSION_HIDDEN_DIMS:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(Config.DROPOUT))
            in_dim = h_dim

        # Final Output Layer (2 targets: formation_energy, bandgap_energy)
        layers.append(nn.Linear(in_dim, len(Config.TARGET_COLS)))

        self.fusion_head = nn.Sequential(*layers)

    def forward(self, x_atomic, x_global, batch_idx):
        """
        Args:
            x_atomic: (Total_Atoms, ATOMIC_INPUT_DIM) - Flattened batch of atoms
            x_global: (Batch_Size, GLOBAL_INPUT_DIM) - Batch of global features
            batch_idx: (Total_Atoms,) - Index mapping each atom to its sample in the batch
        """
        # 1. Process Atomic Features
        h_atomic = self.atomic_stream(x_atomic)  # (Total_Atoms, ATOMIC_HIDDEN_DIM)

        # 2. Dual Pooling (Aggregation)
        # Determine batch size from global features to ensure correct dimension
        batch_size = x_global.size(0)

        # Mean Pooling: Average atomic embeddings for each crystal
        h_mean = scatter_mean(h_atomic, batch_idx, dim=0, dim_size=batch_size)

        # Max Pooling: Max atomic embeddings for each crystal
        # scatter_max returns (values, indices), we only need values
        h_max, _ = scatter_max(h_atomic, batch_idx, dim=0, dim_size=batch_size)

        # 3. Process Global Features
        h_global = self.global_stream(x_global)  # (Batch_Size, GLOBAL_HIDDEN_DIM)

        # 4. Late Fusion
        # Concatenate [Atomic_Mean, Atomic_Max, Global]
        h_fusion = torch.cat([h_mean, h_max, h_global], dim=1)

        # 5. Prediction
        out = self.fusion_head(h_fusion)

        return out
