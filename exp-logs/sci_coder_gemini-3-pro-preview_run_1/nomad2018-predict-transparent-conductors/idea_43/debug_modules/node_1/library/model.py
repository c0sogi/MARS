import torch
import torch.nn as nn
from torch_scatter import scatter
from library.config import Config


class AtomicStream(nn.Module):
    """
    Distortion-Aware Point Processor.
    Processes individual atomic features using a Wide MLP with regularization.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
        super(AtomicStream, self).__init__()

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

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class GlobalStream(nn.Module):
    """
    Anisotropic Physics Context Processor.
    Processes global crystal features using a High-Capacity MLP.
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

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class AMSP_DS_Net(nn.Module):
    """
    Anisotropic Multi-Scale Physics-Aware Deep Sets Network.

    Architecture:
    1. Atomic Stream: Processes sparse atomic features.
    2. Aggregation: Dual Pooling (Mean + Max) of atomic embeddings.
    3. Global Stream: Processes global features.
    4. Late Fusion: Concatenates aggregated atomic and global embeddings.
    5. Regressor: Predicts targets.
    """

    def __init__(
        self,
        atom_input_dim=17,
        global_input_dim=18,
        atomic_hidden_dim=Config.ATOMIC_HIDDEN_DIM,
        atomic_layers=Config.ATOMIC_LAYERS,
        global_hidden_dim=Config.GLOBAL_HIDDEN_DIM,
        global_layers=Config.GLOBAL_LAYERS,
        fusion_hidden_dim=Config.FUSION_HIDDEN_DIM,
        dropout_rate=Config.DROPOUT_RATE,
    ):
        super(AMSP_DS_Net, self).__init__()

        # 1. Atomic Stream
        self.atomic_stream = AtomicStream(
            input_dim=atom_input_dim,
            hidden_dim=atomic_hidden_dim,
            num_layers=atomic_layers,
            dropout_rate=dropout_rate,
        )

        # 2. Global Stream
        self.global_stream = GlobalStream(
            input_dim=global_input_dim,
            hidden_dim=global_hidden_dim,
            num_layers=global_layers,
            dropout_rate=dropout_rate,
        )

        # 3. Fusion Head
        # Input dim = (Atomic Mean + Atomic Max) + Global
        fusion_input_dim = (atomic_hidden_dim * 2) + global_hidden_dim

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim // 2, 2),  # Output: formation_energy, bandgap
        )

    def forward(self, atomic_features, batch_indices, global_features):
        """
        Forward pass.

        Args:
            atomic_features: (N_total_atoms, atom_input_dim)
            batch_indices: (N_total_atoms,) mapping atoms to batch index
            global_features: (Batch_Size, global_input_dim)

        Returns:
            predictions: (Batch_Size, 2)
        """
        # 1. Process Atomic Features
        atomic_emb = self.atomic_stream(
            atomic_features
        )  # (N_total_atoms, atomic_hidden_dim)

        # 2. Dual Pooling (Deep Sets Aggregation)
        # Global Mean Pooling
        # dim=0 is the dimension to reduce
        # index is batch_indices
        # dim_size is the batch size (inferred or explicit)
        batch_size = global_features.size(0)

        mean_pool = scatter(
            atomic_emb, batch_indices, dim=0, dim_size=batch_size, reduce="mean"
        )
        max_pool = scatter(
            atomic_emb, batch_indices, dim=0, dim_size=batch_size, reduce="max"
        )

        # 3. Process Global Features
        global_emb = self.global_stream(
            global_features
        )  # (Batch_Size, global_hidden_dim)

        # 4. Late Fusion
        # Concatenate: [Mean_Pool, Max_Pool, Global_Emb]
        fused = torch.cat([mean_pool, max_pool, global_emb], dim=1)

        # 5. Regression
        output = self.fusion_head(fused)

        return output
