import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max
from library.config import Config


class AtomicStream(nn.Module):
    """
    Processes local atomic features using a wide MLP with regularization.
    Projects atomic features into a high-dimensional latent space.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
        super(AtomicStream, self).__init__()

        layers = []
        # Input layer projection
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        # Hidden layers (Immediate Expansion / Wide MLP)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        # Final linear projection to embedding space (no activation)
        layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class GlobalStream(nn.Module):
    """
    Processes global crystal features (strain, physics, lattice info).
    """

    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
        super(GlobalStream, self).__init__()

        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class AMSA_DS(nn.Module):
    """
    Aligned Multi-Scale Anisotropic Deep Sets (AMSA-DS).

    Architecture:
    1. Atomic Stream: Processes per-atom features (Multi-scale Packing Ratios, Contexts).
    2. Global Stream: Processes per-crystal features (Angular Distortion, Physics).
    3. Aggregation: Dual Pooling (Mean + Max) of atomic embeddings.
    4. Fusion: Concatenation of aggregated atomic and global embeddings.
    5. Head: Final regression MLP.
    """

    def __init__(self):
        super(AMSA_DS, self).__init__()

        # Hyperparameters from Config
        atomic_input_dim = Config.ATOMIC_FEATURE_DIM
        global_input_dim = Config.GLOBAL_FEATURE_DIM

        atomic_hidden = Config.ATOMIC_HIDDEN_DIM
        atomic_layers = Config.ATOMIC_LAYERS

        global_hidden = Config.GLOBAL_HIDDEN_DIM
        global_layers = Config.GLOBAL_LAYERS

        fusion_hidden = Config.FUSION_HIDDEN_DIM
        dropout = Config.DROPOUT_RATE

        # 1. Atomic Stream
        self.atomic_stream = AtomicStream(
            atomic_input_dim, atomic_hidden, atomic_layers, dropout
        )

        # 2. Global Stream
        self.global_stream = GlobalStream(
            global_input_dim, global_hidden, global_layers, dropout
        )

        # 3. Fusion Head
        # Input dim = (2 * atomic_hidden) + global_hidden
        # 2 * atomic because of Dual Pooling (Mean + Max)
        fusion_input_dim = (2 * atomic_hidden) + global_hidden

        self.regressor = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.ReLU(),
            nn.Linear(fusion_hidden // 2, 2),  # 2 Targets: Formation Energy, Bandgap
        )

    def forward(self, atomic_features, global_features, batch_indices, batch_ids=None):
        """
        Args:
            atomic_features (Tensor): (Total_Atoms, Atomic_Dim)
            global_features (Tensor): (Batch_Size, Global_Dim)
            batch_indices (Tensor): (Total_Atoms,) mapping atoms to crystal index in batch (0..B-1)
            batch_ids (Tensor): (Batch_Size,) unused in forward pass

        Returns:
            output (Tensor): (Batch_Size, 2) predicted values
        """
        # 1. Process Atomic Features
        atomic_emb = self.atomic_stream(atomic_features)

        # 2. Aggregation (Dual Pooling)
        # Determine batch size from global features
        batch_size = global_features.shape[0]

        # Global Mean Pooling
        # scatter_mean(src, index, dim, dim_size)
        mean_pool = scatter_mean(atomic_emb, batch_indices, dim=0, dim_size=batch_size)

        # Global Max Pooling
        # scatter_max returns (values, indices)
        max_pool, _ = scatter_max(atomic_emb, batch_indices, dim=0, dim_size=batch_size)

        # Fix for empty graphs: Mask max_pool where count is 0
        ones = torch.ones(
            batch_indices.size(0),
            device=batch_indices.device,
            dtype=atomic_emb.dtype,
        ).unsqueeze(1)
        counts = scatter_add(ones, batch_indices, dim=0, dim_size=batch_size)
        mask = (counts > 0).float()
        max_pool = max_pool * mask

        # Concatenate pooled features
        pooled_atomic = torch.cat([mean_pool, max_pool], dim=1)

        # 3. Process Global Features
        global_emb = self.global_stream(global_features)

        # 4. Late Fusion
        fusion_vec = torch.cat([pooled_atomic, global_emb], dim=1)

        # 5. Regression
        output = self.regressor(fusion_vec)

        return output
