import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max
from library import config


class AtomicStream(nn.Module):
    """
    Chemically-Aware Point Processor.
    Processes individual atomic features including one-hot encoding,
    centered coordinates, and chemically-resolved inverse proximity.
    """

    def __init__(self):
        super().__init__()
        input_dim = config.ATOMIC_INPUT_DIM
        hidden_dim = config.ATOMIC_HIDDEN_DIM
        output_dim = hidden_dim // 2  # Project to 256 dims before pooling

        # Wide MLP with Immediate Expansion and Regularization
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            # Final projection to embedding space (no activation)
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class GlobalStream(nn.Module):
    """
    Thermodynamic Context Processor.
    Processes macroscopic features like lattice vectors, angles, volume, and density.
    """

    def __init__(self):
        super().__init__()
        input_dim = config.GLOBAL_INPUT_DIM
        hidden_dim = config.GLOBAL_HIDDEN_DIM
        output_dim = hidden_dim // 2  # Project to 128 dims

        # High-Capacity MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
        )

    def forward(self, x):
        return self.mlp(x)


class IDCR_WDS_Model(nn.Module):
    """
    Inverse-Distance Chemically-Resolved Wide Deep Sets (IDCR-WDS).

    Architecture:
    1. Atomic features -> AtomicStream -> Dual Pooling (Mean+Max)
    2. Global features -> GlobalStream
    3. Concatenation -> Fusion Head -> Regressor
    """

    def __init__(self):
        super().__init__()
        self.atomic_stream = AtomicStream()
        self.global_stream = GlobalStream()

        # Calculate dimensions for the fusion head
        atom_out_dim = config.ATOMIC_HIDDEN_DIM // 2
        global_out_dim = config.GLOBAL_HIDDEN_DIM // 2

        # Dual pooling concatenates Mean and Max vectors
        fusion_input_dim = (2 * atom_out_dim) + global_out_dim

        # Regression Head
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(256, 128),
            nn.ReLU(),
            # Output layer: 2 targets (formation_energy, bandgap_energy)
            nn.Linear(128, 2),
        )

    def forward(self, atomic_features, batch_indices, global_features):
        """
        Forward pass.

        Args:
            atomic_features (Tensor): (Total_Atoms, Atom_Feat_Dim)
            batch_indices (Tensor): (Total_Atoms,) mapping atoms to crystal index
            global_features (Tensor): (Batch_Size, Global_Feat_Dim)

        Returns:
            Tensor: (Batch_Size, 2) predictions
        """
        # 1. Process Atomic Stream
        atom_emb = self.atomic_stream(atomic_features)

        # 2. Aggregation (Dual Pooling)
        # We use the batch size from global_features to ensure correct dimension
        # even if the last crystal has no atoms (theoretical edge case)
        batch_size = global_features.size(0)

        # Global Mean Pooling
        pool_mean = scatter_mean(atom_emb, batch_indices, dim=0, dim_size=batch_size)

        # Global Max Pooling
        # scatter_max returns (values, indices), we only need values
        pool_max, _ = scatter_max(atom_emb, batch_indices, dim=0, dim_size=batch_size)

        # Handle potential -inf from scatter_max for empty sets by zeroing them
        # (Though in this dataset, crystals are not empty)
        pool_max = torch.nan_to_num(pool_max, nan=0.0, posinf=0.0, neginf=0.0)

        # Concatenate pooled representations
        atom_agg = torch.cat([pool_mean, pool_max], dim=1)

        # 3. Process Global Stream
        global_emb = self.global_stream(global_features)

        # 4. Late Fusion
        fusion_vec = torch.cat([atom_agg, global_emb], dim=1)

        # 5. Regression
        output = self.fusion_head(fusion_vec)

        return output

    def configure_optimizers(self):
        """
        Configures the optimizer and scheduler based on the strategy.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
        )

        # Cite debug_lesson_2: Remove Deprecated `verbose` Argument from PyTorch Scheduler Constructors
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        return optimizer, scheduler
