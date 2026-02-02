import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max
from library.config import (
    ATOMIC_INPUT_DIM,
    GLOBAL_INPUT_DIM,
    ATOMIC_HIDDEN_DIM,
    ATOMIC_LAYERS,
    GLOBAL_HIDDEN_DIM,
    GLOBAL_LAYERS,
    FUSION_HIDDEN_DIM,
    FUSION_LAYERS,
    DROPOUT_RATE,
    NUM_TARGETS,
)


class AtomicStream(nn.Module):
    """
    Chemically-Split Point Processor.
    Processes per-atom features using a Wide MLP with regularization.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        layers = []

        # Initial expansion layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Final linear projection to embedding space (no activation)
        layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GlobalStream(nn.Module):
    """
    Thermodynamic Context Processor.
    Processes crystal-level features.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        layers = []

        # Initial layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Final linear projection
        layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CSNWDS(nn.Module):
    """
    Chemically-Split Neighborhood Wide Deep Sets.

    Architecture:
    1. Atomic Stream: Processes atomic features (One-hot + Coords + Split Distances).
    2. Global Stream: Processes global lattice/composition features.
    3. Aggregation: Dual Pooling (Mean + Max) of atomic embeddings.
    4. Fusion: Concatenates pooled atomic and global embeddings.
    5. Head: Predicts targets.
    """

    def __init__(self):
        super().__init__()

        # 1. Atomic Stream
        self.atomic_stream = AtomicStream(
            input_dim=ATOMIC_INPUT_DIM,
            hidden_dim=ATOMIC_HIDDEN_DIM,
            num_layers=ATOMIC_LAYERS,
            dropout=DROPOUT_RATE,
        )

        # 2. Global Stream
        self.global_stream = GlobalStream(
            input_dim=GLOBAL_INPUT_DIM,
            hidden_dim=GLOBAL_HIDDEN_DIM,
            num_layers=GLOBAL_LAYERS,
            dropout=DROPOUT_RATE,
        )

        # 3. Fusion Head
        # Input dim = (Mean Pool) + (Max Pool) + (Global Emb)
        fusion_input_dim = (2 * ATOMIC_HIDDEN_DIM) + GLOBAL_HIDDEN_DIM

        layers = []
        layers.append(nn.Linear(fusion_input_dim, FUSION_HIDDEN_DIM))
        layers.append(nn.BatchNorm1d(FUSION_HIDDEN_DIM))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(DROPOUT_RATE))

        for _ in range(FUSION_LAYERS - 1):
            layers.append(nn.Linear(FUSION_HIDDEN_DIM, FUSION_HIDDEN_DIM))
            layers.append(nn.BatchNorm1d(FUSION_HIDDEN_DIM))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(DROPOUT_RATE))

        # Output layer (2 targets)
        layers.append(nn.Linear(FUSION_HIDDEN_DIM, NUM_TARGETS))

        self.fusion_head = nn.Sequential(*layers)

    def forward(self, atomic_features, batch_index, global_features):
        """
        Args:
            atomic_features (Tensor): (Total_Atoms, ATOMIC_INPUT_DIM)
            batch_index (Tensor): (Total_Atoms,) mapping atoms to batch indices
            global_features (Tensor): (Batch_Size, GLOBAL_INPUT_DIM)

        Returns:
            Tensor: (Batch_Size, NUM_TARGETS)
        """
        # 1. Process Atomic Features
        atom_emb = self.atomic_stream(
            atomic_features
        )  # (Total_Atoms, ATOMIC_HIDDEN_DIM)

        # 2. Aggregate Atomic Embeddings (Dual Pooling)
        batch_size = global_features.shape[0]

        # Mean Pooling
        mean_pool = scatter_mean(
            atom_emb, batch_index, dim=0, dim_size=batch_size
        )  # (B, H_a)

        # Max Pooling (scatter_max returns values, indices)
        max_pool, _ = scatter_max(
            atom_emb, batch_index, dim=0, dim_size=batch_size
        )  # (B, H_a)

        # 3. Process Global Features
        glob_emb = self.global_stream(global_features)  # (B, H_g)

        # 4. Late Fusion
        combined = torch.cat([mean_pool, max_pool, glob_emb], dim=1)  # (B, 2*H_a + H_g)

        # 5. Prediction
        output = self.fusion_head(combined)  # (B, 2)

        return output
