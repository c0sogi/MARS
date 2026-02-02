import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max
from library.config import (
    ATOMIC_INPUT_DIM,
    ATOMIC_HIDDEN_DIM,
    ATOMIC_DROPOUT,
    GLOBAL_INPUT_DIM,
    GLOBAL_HIDDEN_DIM,
    GLOBAL_DROPOUT,
    FUSION_HIDDEN_DIM,
    OUTPUT_DIM,
)


class AtomicStream(nn.Module):
    """
    Wide Point Processor for atomic features.

    Processes per-atom features including one-hot encoding, centered coordinates,
    nearest neighbor distance, and local potential proxy.
    Uses a wide MLP with Batch Normalization and Dropout.
    """

    def __init__(
        self,
        input_dim=ATOMIC_INPUT_DIM,
        hidden_dim=ATOMIC_HIDDEN_DIM,
        output_dim=256,
        dropout=ATOMIC_DROPOUT,
    ):
        super().__init__()

        # Wide MLP layers
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Final linear projection to embedding space (no activation)
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Atomic features (Total_Atoms, ATOMIC_INPUT_DIM)
        Returns:
            Tensor: Atomic embeddings (Total_Atoms, output_dim)
        """
        return self.net(x)


class GlobalStream(nn.Module):
    """
    Thermodynamic Context Encoder.

    Processes macroscopic features like lattice parameters, volume, density,
    and stoichiometry using a high-capacity MLP.
    """

    def __init__(
        self,
        input_dim=GLOBAL_INPUT_DIM,
        hidden_dim=GLOBAL_HIDDEN_DIM,
        dropout=GLOBAL_DROPOUT,
    ):
        super().__init__()

        # Deep MLP for global context
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Global features (Batch_Size, GLOBAL_INPUT_DIM)
        Returns:
            Tensor: Global embeddings (Batch_Size, hidden_dim)
        """
        return self.net(x)


class FusionHead(nn.Module):
    """
    Late Fusion and Regression Head.

    Aggregates atomic embeddings via Dual Pooling (Mean + Max), concatenates
    with global embeddings, and predicts targets.
    """

    def __init__(
        self,
        atomic_emb_dim=256,
        global_emb_dim=GLOBAL_HIDDEN_DIM,
        hidden_dim=FUSION_HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
    ):
        super().__init__()

        # Input dim = (Mean Pool) + (Max Pool) + (Global Context)
        fusion_input_dim = (atomic_emb_dim * 2) + global_emb_dim

        self.net = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, atomic_emb, batch_indices, global_emb):
        """
        Args:
            atomic_emb (Tensor): Per-atom embeddings (Total_Atoms, atomic_emb_dim)
            batch_indices (Tensor): Batch index for each atom (Total_Atoms,)
            global_emb (Tensor): Global embeddings (Batch_Size, global_emb_dim)
        Returns:
            Tensor: Predictions (Batch_Size, output_dim)
        """
        # Dual Pooling: Global Mean + Global Max
        # scatter_mean: (Batch_Size, atomic_emb_dim)
        mean_pool = scatter_mean(atomic_emb, batch_indices, dim=0)

        # scatter_max returns (values, indices), we only need values
        # Initialize with very small number for max pooling stability if needed,
        # but scatter_max usually handles it.
        max_pool, _ = scatter_max(atomic_emb, batch_indices, dim=0)

        # Handle cases where a batch might have missing indices (though unlikely in this pipeline)
        # Ensure dimensions match global_emb
        if mean_pool.shape[0] != global_emb.shape[0]:
            # This theoretically shouldn't happen if batch_indices covers 0 to B-1
            pass

        # Concatenate: [Mean, Max, Global]
        combined = torch.cat([mean_pool, max_pool, global_emb], dim=1)

        return self.net(combined)


class PCWDSModel(nn.Module):
    """
    Potential-Calibrated Wide Deep Sets (PC-WDS) Model.

    Combines AtomicStream, GlobalStream, and FusionHead.
    """

    def __init__(self):
        super().__init__()

        # Atomic Stream projecting to 256 dim embedding
        self.atomic_stream = AtomicStream(
            input_dim=ATOMIC_INPUT_DIM,
            hidden_dim=ATOMIC_HIDDEN_DIM,
            output_dim=256,
            dropout=ATOMIC_DROPOUT,
        )

        # Global Stream projecting to GLOBAL_HIDDEN_DIM (256)
        self.global_stream = GlobalStream(
            input_dim=GLOBAL_INPUT_DIM,
            hidden_dim=GLOBAL_HIDDEN_DIM,
            dropout=GLOBAL_DROPOUT,
        )

        # Fusion Head
        self.fusion_head = FusionHead(
            atomic_emb_dim=256,
            global_emb_dim=GLOBAL_HIDDEN_DIM,
            hidden_dim=FUSION_HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
        )

    def forward(self, atom_x, batch_indices, global_x):
        """
        Forward pass.

        Args:
            atom_x (Tensor): Atomic features (Total_Atoms, 9)
            batch_indices (Tensor): Batch mapping (Total_Atoms,)
            global_x (Tensor): Global features (Batch_Size, 12)

        Returns:
            Tensor: Predicted energies (Batch_Size, 2)
        """
        # 1. Process Atomic Stream
        atomic_emb = self.atomic_stream(atom_x)

        # 2. Process Global Stream
        global_emb = self.global_stream(global_x)

        # 3. Fuse and Regress
        output = self.fusion_head(atomic_emb, batch_indices, global_emb)

        return output
