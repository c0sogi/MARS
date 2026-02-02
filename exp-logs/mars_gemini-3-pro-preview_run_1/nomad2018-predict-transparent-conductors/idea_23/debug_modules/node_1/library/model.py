import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max
from library.config import (
    ATOM_INPUT_DIM,
    ATOMIC_HIDDEN_DIM,
    GLOBAL_INPUT_DIM,
    GLOBAL_HIDDEN_DIM,
    DROPOUT,
)


class AtomicEncoder(nn.Module):
    """
    Chemically-Contextualized Point Processor.
    Processes individual atomic feature vectors using a Wide MLP.
    """

    def __init__(self, input_dim, hidden_dim, dropout):
        super(AtomicEncoder, self).__init__()

        # Immediate Expansion to wide hidden dimension
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Final projection to embedding space (no activation/BN at the very end of encoder usually,
            # but keeping consistent with "Deep Sets" philosophy often implies mapping to latent space)
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class GlobalEncoder(nn.Module):
    """
    Encodes macroscopic thermodynamic context features.
    """

    def __init__(self, input_dim, hidden_dim, dropout):
        super(GlobalEncoder, self).__init__()

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


class CCWDS(nn.Module):
    """
    Chemically-Contextualized Wide Deep Sets.

    Architecture:
    1. Atomic Stream: Processes atom-level features (Self, Spatial, NN Dist, NN ID).
       Aggregates via Dual Pooling (Mean + Max).
    2. Global Stream: Processes crystal-level features (Lattice, Vol, Density, Stoichiometry).
    3. Fusion Head: Concatenates streams and predicts targets.
    """

    def __init__(self):
        super(CCWDS, self).__init__()

        # 1. Atomic Stream
        self.atomic_encoder = AtomicEncoder(
            input_dim=ATOM_INPUT_DIM, hidden_dim=ATOMIC_HIDDEN_DIM, dropout=DROPOUT
        )

        # 2. Global Stream
        self.global_encoder = GlobalEncoder(
            input_dim=GLOBAL_INPUT_DIM, hidden_dim=GLOBAL_HIDDEN_DIM, dropout=DROPOUT
        )

        # 3. Fusion Head
        # Input: (Atomic_Mean + Atomic_Max) + Global
        fusion_input_dim = (ATOMIC_HIDDEN_DIM * 2) + GLOBAL_HIDDEN_DIM

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 2),  # Predicts formation_energy and bandgap_energy
        )

    def forward(self, batch_atomic, batch_indices, batch_global):
        """
        Args:
            batch_atomic: (Sum_N_atoms, ATOM_INPUT_DIM)
            batch_indices: (Sum_N_atoms,) - Sample index for each atom
            batch_global: (B, GLOBAL_INPUT_DIM)

        Returns:
            output: (B, 2)
        """
        # Determine batch size from global features (which are always present)
        batch_size = batch_global.size(0)

        # --- Atomic Stream ---
        # Encode individual atoms
        atom_embeddings = self.atomic_encoder(
            batch_atomic
        )  # (Sum_N, ATOMIC_HIDDEN_DIM)

        # Dual Pooling (Scatter Mean and Max)
        # dim=0 is the dimension to reduce
        # index=batch_indices maps atoms to batch elements
        # Fix for Lesson ID: debug_lesson_11
        # Explicitly pass dim_size to ensure output has size B even if some indices are missing
        pooled_mean = scatter_mean(
            atom_embeddings, batch_indices, dim=0, dim_size=batch_size
        )  # (B, ATOMIC_HIDDEN_DIM)
        pooled_max, _ = scatter_max(
            atom_embeddings, batch_indices, dim=0, dim_size=batch_size
        )  # (B, ATOMIC_HIDDEN_DIM)

        # --- Global Stream ---
        global_embeddings = self.global_encoder(batch_global)  # (B, GLOBAL_HIDDEN_DIM)

        # --- Fusion ---
        # Concatenate: [Mean_Pool, Max_Pool, Global]
        combined = torch.cat(
            [pooled_mean, pooled_max, global_embeddings], dim=1
        )  # (B, 1280)

        # Predict
        output = self.fusion_head(combined)  # (B, 2)

        return output
