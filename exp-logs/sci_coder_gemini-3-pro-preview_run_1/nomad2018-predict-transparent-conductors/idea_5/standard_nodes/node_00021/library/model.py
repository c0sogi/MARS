import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    NUM_GLOBAL_FEATURES,
    GLOBAL_HIDDEN_DIM,
    GLOBAL_LATENT_DIM,
    ATOMIC_INPUT_DIM,
    ATOMIC_HIDDEN_DIM,
    LATENT_DIM,
    HEAD_HIDDEN_DIM,
    DROPOUT_RATE,
)


class GlobalEncoder(nn.Module):
    """
    Encodes macroscopic features (lattice, volume, density, composition).
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_GLOBAL_FEATURES, GLOBAL_HIDDEN_DIM),
            nn.BatchNorm1d(GLOBAL_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(GLOBAL_HIDDEN_DIM, GLOBAL_HIDDEN_DIM),
            nn.BatchNorm1d(GLOBAL_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(GLOBAL_HIDDEN_DIM, GLOBAL_LATENT_DIM),
            # Linear output for fusion (Cite 18)
        )

    def forward(self, global_features):
        return self.net(global_features)


class AtomicEncoder(nn.Module):
    """
    Processes atomic features (One-Hot + Coords + Fingerprints).
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ATOMIC_INPUT_DIM, ATOMIC_HIDDEN_DIM),
            nn.BatchNorm1d(ATOMIC_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(ATOMIC_HIDDEN_DIM, ATOMIC_HIDDEN_DIM),
            nn.BatchNorm1d(ATOMIC_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(ATOMIC_HIDDEN_DIM, LATENT_DIM),
            # Linear output before pooling (Cite 18)
        )

    def forward(self, x):
        return self.net(x)


class DualPoolingAggregator(nn.Module):
    """
    Performs Mean and Max Pooling (Cite 5).
    """

    def __init__(self):
        super().__init__()

    def forward(self, atomic_embeddings, mask):
        """
        Args:
            atomic_embeddings: (Batch, MaxAtoms, LatentDim)
            mask: (Batch, MaxAtoms) - True for real atoms
        """
        # Mask out padding
        mask_expanded = mask.unsqueeze(-1)  # (Batch, MaxAtoms, 1)

        # 1. Mean Pooling
        # Zero out padding
        masked_embeddings_zero = atomic_embeddings * mask_expanded.float()
        sum_embeddings = torch.sum(masked_embeddings_zero, dim=1)
        counts = mask.sum(dim=1, keepdim=True).float()
        mean_pool = sum_embeddings / (counts + 1e-6)

        # 2. Max Pooling
        # Set padding to large negative
        masked_embeddings_neg = atomic_embeddings.masked_fill(~mask_expanded, -1e9)
        max_pool, _ = torch.max(masked_embeddings_neg, dim=1)

        return torch.cat([mean_pool, max_pool], dim=1)


class DualStreamModel(nn.Module):
    """
    Dual-Stream Deep Sets with Late Fusion (Cite 19).
    """

    def __init__(self):
        super().__init__()
        self.global_encoder = GlobalEncoder()
        self.atomic_encoder = AtomicEncoder()
        self.aggregator = DualPoolingAggregator()

        # Input to head is: (LatentDim * 2 from pooling) + GlobalLatentDim
        fusion_dim = (LATENT_DIM * 2) + GLOBAL_LATENT_DIM

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, HEAD_HIDDEN_DIM),
            nn.BatchNorm1d(HEAD_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HEAD_HIDDEN_DIM, HEAD_HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(HEAD_HIDDEN_DIM // 2, 2),
        )

    def forward(self, global_features, atomic_features, mask):
        # 1. Global Stream
        global_vec = self.global_encoder(global_features)

        # 2. Atomic Stream
        batch_size, max_atoms, _ = atomic_features.shape
        flat_input = atomic_features.view(-1, ATOMIC_INPUT_DIM)
        flat_embeddings = self.atomic_encoder(flat_input)
        atomic_embeddings = flat_embeddings.view(batch_size, max_atoms, LATENT_DIM)

        # 3. Aggregation
        atomic_vec = self.aggregator(atomic_embeddings, mask)

        # 4. Late Fusion (Cite 19)
        fused_vec = torch.cat([atomic_vec, global_vec], dim=1)

        # 5. Prediction
        output = self.head(fused_vec)

        return output
