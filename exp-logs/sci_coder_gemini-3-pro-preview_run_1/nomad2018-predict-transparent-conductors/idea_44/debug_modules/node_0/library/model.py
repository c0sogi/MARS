import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max
from library.config import Config


class AtomicStream(nn.Module):
    """
    Distortion-Aware Point Processor for atomic features.
    Processes the 17-dimensional atomic feature vectors using a wide MLP
    with Batch Normalization and Dropout.
    """

    def __init__(
        self,
        input_dim=17,
        hidden_dim=Config.ATOMIC_HIDDEN_DIM,
        output_dim=Config.LATENT_DIM,
        num_layers=Config.ATOMIC_LAYERS,
        dropout=Config.DROPOUT,
    ):
        super(AtomicStream, self).__init__()

        layers = []

        # Input layer
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

        # Output projection (Linear, no activation)
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GlobalStream(nn.Module):
    """
    Anisotropic Physics Context encoder.
    Processes the 19-dimensional global feature vectors using a high-capacity MLP.
    """

    def __init__(
        self,
        input_dim=19,
        hidden_dim=Config.GLOBAL_HIDDEN_DIM,
        output_dim=Config.LATENT_DIM,
        num_layers=Config.GLOBAL_LAYERS,
        dropout=Config.DROPOUT,
    ):
        super(GlobalStream, self).__init__()

        layers = []

        # Input layer
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

        # Output projection
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.BatchNorm1d(output_dim))  # Normalize embedding before fusion
        layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AMSP_DS_Net(nn.Module):
    """
    Main Anisotropic Multi-Scale Physics-Aware Deep Sets Network.

    Architecture:
    1. Atomic Stream: Processes atom-level features.
    2. Dual Pooling: Aggregates atomic embeddings via Mean and Max pooling.
    3. Global Stream: Processes crystal-level features.
    4. Fusion: Concatenates pooled atomic embeddings and global embeddings.
    5. Regressor: Predicts formation energy and bandgap.
    """

    def __init__(self):
        super(AMSP_DS_Net, self).__init__()

        # Feature dimensions based on extraction logic
        self.atomic_input_dim = 17
        self.global_input_dim = 19

        # 1. Atomic Stream
        self.atomic_stream = AtomicStream(
            input_dim=self.atomic_input_dim,
            hidden_dim=Config.ATOMIC_HIDDEN_DIM,
            output_dim=Config.LATENT_DIM,
            num_layers=Config.ATOMIC_LAYERS,
            dropout=Config.DROPOUT,
        )

        # 2. Global Stream
        self.global_stream = GlobalStream(
            input_dim=self.global_input_dim,
            hidden_dim=Config.GLOBAL_HIDDEN_DIM,
            output_dim=Config.LATENT_DIM,
            num_layers=Config.GLOBAL_LAYERS,
            dropout=Config.DROPOUT,
        )

        # 3. Fusion Head
        # Input: (Mean Pool + Max Pool) + Global Embedding
        # Size: LATENT_DIM + LATENT_DIM + LATENT_DIM = 3 * LATENT_DIM
        fusion_input_dim = 3 * Config.LATENT_DIM

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(256, 2),  # Output: [formation_energy, bandgap]
        )

    def forward(self, atomic_features, global_features, batch_indices):
        """
        Forward pass.

        Args:
            atomic_features (Tensor): (N_total_atoms, 17)
            global_features (Tensor): (Batch_size, 19)
            batch_indices (Tensor): (N_total_atoms,) mapping atoms to batch index

        Returns:
            Tensor: (Batch_size, 2) predictions
        """
        # 1. Process Atomic Features
        # Shape: (N_total_atoms, LATENT_DIM)
        atom_embeddings = self.atomic_stream(atomic_features)

        # 2. Dual Pooling (Scatter Aggregation)
        # Global Mean Pooling
        # Shape: (Batch_size, LATENT_DIM)
        mean_pool = scatter_mean(atom_embeddings, batch_indices, dim=0)

        # Global Max Pooling
        # Shape: (Batch_size, LATENT_DIM)
        # scatter_max returns (values, indices), we only need values
        max_pool, _ = scatter_max(atom_embeddings, batch_indices, dim=0)

        # 3. Process Global Features
        # Shape: (Batch_size, LATENT_DIM)
        global_embeddings = self.global_stream(global_features)

        # 4. Late Fusion
        # Concatenate: [Mean, Max, Global]
        # Shape: (Batch_size, 3 * LATENT_DIM)
        fused = torch.cat([mean_pool, max_pool, global_embeddings], dim=1)

        # 5. Regression
        # Shape: (Batch_size, 2)
        output = self.fusion_head(fused)

        return output
