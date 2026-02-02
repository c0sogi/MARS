import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool

from library.config import (
    ATOMIC_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    HIDDEN_DIM,
    LATENT_DIM,
    DROPOUT_RATE,
    NUM_TARGETS,
    USE_BATCH_NORM,
    SEED,
)

# Set seeds for reproducibility
torch.manual_seed(SEED)


class AtomicStream(nn.Module):
    """
    Bond-Aware Point Processor.
    Wide MLP to encode atomic features into a high-dimensional latent space.
    """

    def __init__(self, input_dim, hidden_dim, dropout_rate, use_batch_norm):
        super(AtomicStream, self).__init__()

        layers = []
        # Layer 1: Expansion
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        # Layer 2: Processing
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        # Layer 3: Deepening
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GlobalStream(nn.Module):
    """
    Disorder-Aware Context Processor.
    High-Capacity MLP to encode global crystal features.
    """

    def __init__(self, input_dim, hidden_dim, dropout_rate, use_batch_norm):
        super(GlobalStream, self).__init__()

        layers = []
        # Layer 1
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        # Layer 2
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BA_ADS_Model(nn.Module):
    """
    Bond-Aware Anisotropic Deep Sets (BA-ADS) Model.
    Combines Atomic Stream (with Dual Pooling) and Global Stream via Late Fusion.
    """

    def __init__(self):
        super(BA_ADS_Model, self).__init__()

        # Atomic Stream: Wide MLP (e.g., 512 units)
        self.atomic_stream = AtomicStream(
            input_dim=ATOMIC_FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            dropout_rate=DROPOUT_RATE,
            use_batch_norm=USE_BATCH_NORM,
        )

        # Global Stream: High-Capacity MLP (e.g., 256 units)
        self.global_stream = GlobalStream(
            input_dim=GLOBAL_FEATURE_DIM,
            hidden_dim=LATENT_DIM,
            dropout_rate=DROPOUT_RATE,
            use_batch_norm=USE_BATCH_NORM,
        )

        # Fusion Head
        # Atomic aggregation: Mean (HIDDEN_DIM) + Max (HIDDEN_DIM)
        # Global embedding: LATENT_DIM
        fusion_input_dim = (HIDDEN_DIM * 2) + LATENT_DIM

        self.regressor = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_TARGETS),
        )

    def forward(self, batch_atomic, batch_index, batch_global):
        """
        Forward pass.

        Args:
            batch_atomic: (Total_Atoms, Atomic_Dim)
            batch_index: (Total_Atoms,) - Crystal index for each atom
            batch_global: (Batch_Size, Global_Dim)
        """
        # 1. Atomic Stream Processing
        atomic_emb = self.atomic_stream(batch_atomic)  # (Total_Atoms, HIDDEN_DIM)

        # 2. Aggregation (Dual Pooling)
        # Global Mean Pooling
        atomic_mean = global_mean_pool(
            atomic_emb, batch_index
        )  # (Batch_Size, HIDDEN_DIM)
        # Global Max Pooling
        atomic_max = global_max_pool(
            atomic_emb, batch_index
        )  # (Batch_Size, HIDDEN_DIM)

        # 3. Global Stream Processing
        global_emb = self.global_stream(batch_global)  # (Batch_Size, LATENT_DIM)

        # 4. Late Fusion
        # Concatenate: [Atomic_Mean, Atomic_Max, Global]
        fusion_vec = torch.cat(
            [atomic_mean, atomic_max, global_emb], dim=1
        )  # (Batch_Size, Fusion_Dim)

        # 5. Final Regression
        output = self.regressor(fusion_vec)  # (Batch_Size, NUM_TARGETS)

        return output
