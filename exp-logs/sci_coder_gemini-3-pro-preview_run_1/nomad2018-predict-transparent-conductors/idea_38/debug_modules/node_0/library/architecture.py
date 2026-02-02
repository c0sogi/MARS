import torch
import torch.nn as nn
from library.config import (
    ATOMIC_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    HIDDEN_DIM_ATOMIC,
    HIDDEN_DIM_GLOBAL,
    FUSION_HIDDEN_DIM,
    DROPOUT_RATE,
)


class AtomicStream(nn.Module):
    """
    Robust Point Processor: A Wide MLP to encode per-atom features.
    Applies Batch Normalization and Dropout after activations.
    """

    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(AtomicStream, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            # Final projection to embedding space (linear)
        )

    def forward(self, x):
        # x shape: (Batch, Max_Atoms, Input_Dim)
        # Reshape for Linear layers: (Batch * Max_Atoms, Input_Dim)
        b, n, d = x.shape
        x_flat = x.view(b * n, d)
        out_flat = self.net(x_flat)
        # Reshape back: (Batch, Max_Atoms, Hidden_Dim)
        return out_flat.view(b, n, -1)


class GlobalStream(nn.Module):
    """
    Structurally-Enhanced Context: High-Capacity MLP for global features.
    """

    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(GlobalStream, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            # Final projection (linear)
        )

    def forward(self, x):
        # x shape: (Batch, Input_Dim)
        return self.net(x)


class SSAWDSModel(nn.Module):
    """
    Structural-Statistics Augmented Wide Deep Sets (SSA-WDS).
    Merges local atomic embeddings (via Dual Pooling) with global structural context.
    """

    def __init__(
        self,
        atomic_input_dim=ATOMIC_FEATURE_DIM,
        global_input_dim=GLOBAL_FEATURE_DIM,
        atomic_hidden_dim=HIDDEN_DIM_ATOMIC,
        global_hidden_dim=HIDDEN_DIM_GLOBAL,
        fusion_hidden_dim=FUSION_HIDDEN_DIM,
        dropout_rate=DROPOUT_RATE,
    ):
        super(SSAWDSModel, self).__init__()

        # 1. Atomic Stream
        self.atomic_stream = AtomicStream(
            atomic_input_dim, atomic_hidden_dim, dropout_rate
        )

        # 2. Global Stream
        self.global_stream = GlobalStream(
            global_input_dim, global_hidden_dim, dropout_rate
        )

        # 3. Fusion Head
        # Input: (Mean_Pool + Max_Pool) + Global_Embedding
        fusion_input_dim = (atomic_hidden_dim * 2) + global_hidden_dim

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden_dim, 2),  # Output: formation_energy, bandgap
        )

    def forward(self, atomic_features, atomic_mask, global_features):
        """
        Args:
            atomic_features: (Batch, Max_Atoms, Atom_Dim)
            atomic_mask: (Batch, Max_Atoms) - 1 for atom, 0 for padding
            global_features: (Batch, Global_Dim)

        Returns:
            predictions: (Batch, 2)
        """
        # --- Atomic Processing ---
        # (Batch, Max_Atoms, Atomic_Hidden)
        atom_embeddings = self.atomic_stream(atomic_features)

        # --- Dual Pooling ---
        # Expand mask for broadcasting: (Batch, Max_Atoms, 1)
        mask_expanded = atomic_mask.unsqueeze(-1)

        # 1. Global Mean Pooling
        # Sum valid embeddings
        sum_embeddings = torch.sum(atom_embeddings * mask_expanded, dim=1)
        # Count valid atoms (avoid div by zero)
        counts = torch.sum(mask_expanded, dim=1)
        counts = torch.clamp(counts, min=1.0)
        mean_pool = sum_embeddings / counts

        # 2. Global Max Pooling
        # Mask padding with a very small number before max
        # Clone to avoid modifying gradients in place incorrectly if reused
        masked_for_max = atom_embeddings.clone()
        masked_for_max = masked_for_max.masked_fill(mask_expanded == 0, -1e9)
        max_pool, _ = torch.max(masked_for_max, dim=1)

        # --- Global Processing ---
        # (Batch, Global_Hidden)
        global_embedding = self.global_stream(global_features)

        # --- Late Fusion ---
        # Concatenate: [Mean_Pool, Max_Pool, Global]
        # Dim: Atomic_Hidden + Atomic_Hidden + Global_Hidden
        fused = torch.cat([mean_pool, max_pool, global_embedding], dim=1)

        # --- Prediction ---
        output = self.fusion_head(fused)

        return output
