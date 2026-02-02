import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max
from library.config import Config


class AtomicEncoder(nn.Module):
    """
    Wide MLP to project atomic features into a high-dimensional latent space.
    Processes each atom independently before aggregation.

    Input: (Total_Atoms, ATOM_INPUT_DIM)
    Output: (Total_Atoms, LATENT_DIM)
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
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
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class GlobalEncoder(nn.Module):
    """
    High-capacity MLP to encode macroscopic properties (lattice, composition, etc.).
    Ensures global context is sufficiently processed before fusion.

    Input: (Batch_Size, GLOBAL_INPUT_DIM)
    Output: (Batch_Size, LATENT_DIM)
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
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
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class APDeepSets(nn.Module):
    """
    Augmented Point-Cloud Deep Sets (AP-DeepSets) Architecture.

    This model integrates:
    1. An Atomic Stream that learns from local atomic environments (identity, coords, neighbors).
    2. A Global Stream that learns from macroscopic unit cell properties.
    3. Dual Pooling (Mean + Max) to capture both average and distinct atomic features.
    4. Late Fusion for final regression.
    """

    def __init__(self):
        super().__init__()

        # Hyperparameters from Config
        atom_in_dim = Config.ATOM_INPUT_DIM
        global_in_dim = Config.GLOBAL_INPUT_DIM
        hidden_dim = Config.HIDDEN_DIM
        latent_dim = Config.LATENT_DIM
        dropout = Config.DROPOUT
        num_targets = Config.NUM_TARGETS

        # 1. Atomic Stream Encoder
        self.atomic_encoder = AtomicEncoder(
            input_dim=atom_in_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            dropout=dropout,
        )

        # 2. Global Stream Encoder
        self.global_encoder = GlobalEncoder(
            input_dim=global_in_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            dropout=dropout,
        )

        # 3. Fusion and Regressor
        # The fusion vector concatenates:
        # - Global Mean of Atomic Embeddings (latent_dim)
        # - Global Max of Atomic Embeddings (latent_dim)
        # - Encoded Global Features (latent_dim)
        # Total Dimension = 3 * latent_dim
        fusion_dim = 3 * latent_dim

        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_targets),
        )

    def forward(self, batch_data):
        """
        Forward pass of the AP-DeepSets model.

        Args:
            batch_data (dict): Dictionary containing:
                - 'atomic_features': Tensor (Total_Atoms, ATOM_INPUT_DIM)
                - 'global_features': Tensor (Batch_Size, GLOBAL_INPUT_DIM)
                - 'batch_indices': Tensor (Total_Atoms,) mapping atoms to batch index

        Returns:
            Tensor: Predictions of shape (Batch_Size, NUM_TARGETS)
        """
        atomic_x = batch_data["atomic_features"]
        global_x = batch_data["global_features"]
        batch_idx = batch_data["batch_indices"]

        batch_size = global_x.size(0)

        # --- Atomic Stream ---
        # 1. Encode per-atom features
        # Shape: (Total_Atoms, latent_dim)
        atom_emb = self.atomic_encoder(atomic_x)

        # 2. Dual Pooling (Global Mean + Global Max)
        # Aggregates variable number of atoms into fixed-size vectors per sample
        # Shape: (Batch_Size, latent_dim)
        mean_pool = scatter_mean(atom_emb, batch_idx, dim=0, dim_size=batch_size)
        max_pool, _ = scatter_max(atom_emb, batch_idx, dim=0, dim_size=batch_size)

        # --- Global Stream ---
        # 3. Encode global context features
        # Shape: (Batch_Size, latent_dim)
        global_emb = self.global_encoder(global_x)

        # --- Fusion ---
        # 4. Concatenate aggregated atomic representations with global context
        # Shape: (Batch_Size, 3 * latent_dim)
        fused_features = torch.cat([mean_pool, max_pool, global_emb], dim=1)

        # --- Prediction ---
        # 5. Final regression
        # Shape: (Batch_Size, NUM_TARGETS)
        predictions = self.regressor(fused_features)

        return predictions
