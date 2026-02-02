import torch
import torch.nn as nn

try:
    from torch_scatter import scatter_mean, scatter_max
except ImportError:
    pass
from library.config import Config


class AtomicStream(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super().__init__()
        # Wide MLP with Immediate Expansion
        # Structure: Linear -> BN -> ReLU -> Dropout
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # Final projection to embedding space (no activation as per strategy)
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class GlobalStream(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super().__init__()
        # High-Capacity MLP for thermodynamic context
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class CRRD_DeepSets(nn.Module):
    def __init__(self):
        super().__init__()

        # Dimensions from Config
        atomic_input_dim = Config.ATOMIC_FEATURE_DIM
        global_input_dim = Config.GLOBAL_FEATURE_DIM

        hidden_atomic = Config.HIDDEN_DIM_ATOMIC
        hidden_global = Config.HIDDEN_DIM_GLOBAL
        embedding_dim = Config.HIDDEN_DIM_FUSION
        dropout = Config.DROPOUT_RATE

        # 1. Atomic Stream (Chemically-Aware Point Processor)
        self.atomic_stream = AtomicStream(
            input_dim=atomic_input_dim,
            hidden_dim=hidden_atomic,
            output_dim=embedding_dim,
            dropout_rate=dropout,
        )

        # 2. Global Stream (Thermodynamic Context)
        self.global_stream = GlobalStream(
            input_dim=global_input_dim,
            hidden_dim=hidden_global,
            output_dim=embedding_dim,
            dropout_rate=dropout,
        )

        # 3. Fusion Head
        # Concatenation of:
        # - Global Embedding (embedding_dim)
        # - Atomic Mean Pooling (embedding_dim)
        # - Atomic Max Pooling (embedding_dim)
        fusion_input_dim = embedding_dim * 3

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),  # Predicts [formation_energy, bandgap_energy]
        )

    def forward(self, atomic_feats, batch_indices, global_feats):
        """
        Args:
            atomic_feats: (N_atoms, 11)
            batch_indices: (N_atoms,) indicating which crystal each atom belongs to
            global_feats: (Batch_Size, 12)
        """
        # --- Atomic Stream ---
        # Project atoms to latent space
        atomic_emb = self.atomic_stream(atomic_feats)  # (N_atoms, embedding_dim)

        # --- Aggregation (Dual Pooling) ---
        # Ensure we scatter into the correct batch size
        batch_size = global_feats.size(0)

        # Mean Pooling
        # (Batch_Size, embedding_dim)
        atomic_mean = scatter_mean(
            atomic_emb, batch_indices, dim=0, dim_size=batch_size
        )

        # Max Pooling
        # scatter_max returns (values, indices)
        atomic_max, _ = scatter_max(
            atomic_emb, batch_indices, dim=0, dim_size=batch_size
        )

        # Handle potential -inf from max pooling if a batch element had 0 atoms
        atomic_max = torch.nan_to_num(atomic_max, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Global Stream ---
        # (Batch_Size, embedding_dim)
        global_emb = self.global_stream(global_feats)

        # --- Late Fusion ---
        # Concatenate all embeddings
        # (Batch_Size, embedding_dim * 3)
        fusion_vec = torch.cat([atomic_mean, atomic_max, global_emb], dim=1)

        # --- Regression ---
        output = self.fusion_head(fusion_vec)

        return output
