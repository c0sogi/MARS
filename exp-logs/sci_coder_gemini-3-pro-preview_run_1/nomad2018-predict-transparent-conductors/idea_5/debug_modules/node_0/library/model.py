import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    NUM_GLOBAL_FEATURES,
    GLOBAL_HIDDEN_DIM,
    GLOBAL_CONTEXT_DIM,
    ATOMIC_INPUT_DIM,
    ATOMIC_HIDDEN_DIM,
    LATENT_DIM,
    HEAD_HIDDEN_DIM,
    DROPOUT_RATE,
)


class GlobalContextEncoder(nn.Module):
    """
    Encodes macroscopic features (lattice, volume, density, composition)
    into a global context vector.
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
            nn.Linear(GLOBAL_HIDDEN_DIM, GLOBAL_CONTEXT_DIM),
            nn.Tanh(),  # Tanh to bound the context signal, often helpful for conditioning
        )

    def forward(self, global_features):
        return self.net(global_features)


class ContextAwareAtomicEncoder(nn.Module):
    """
    Processes atomic features. The input is expected to be the concatenation
    of the base atomic features and the expanded global context vector.
    """

    def __init__(self):
        super().__init__()
        # Wide MLP structure
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
            # No activation at the end to preserve full vector space for pooling
        )

    def forward(self, x):
        # x shape: (Batch * MaxAtoms, ATOMIC_INPUT_DIM)
        # We process flattened batch for BatchNorm efficiency
        return self.net(x)


class AttentionAggregator(nn.Module):
    """
    Performs Attention-Weighted Mean Pooling and Global Max Pooling.
    """

    def __init__(self):
        super().__init__()
        # Attention scorer: maps latent dim to a scalar score
        self.attention_net = nn.Sequential(
            nn.Linear(LATENT_DIM, 64), nn.Tanh(), nn.Linear(64, 1)
        )

    def forward(self, atomic_embeddings, mask):
        """
        Args:
            atomic_embeddings: (Batch, MaxAtoms, LatentDim)
            mask: (Batch, MaxAtoms) - True for real atoms, False for padding
        """
        # 1. Compute Attention Scores
        # scores: (Batch, MaxAtoms, 1)
        scores = self.attention_net(atomic_embeddings)

        # Mask out padding (set to large negative number before softmax)
        # mask is (Batch, MaxAtoms), unsqueeze to broadcast
        mask_expanded = mask.unsqueeze(-1)  # (Batch, MaxAtoms, 1)
        scores = scores.masked_fill(~mask_expanded, -1e9)

        # Compute weights
        alpha = F.softmax(scores, dim=1)  # (Batch, MaxAtoms, 1)

        # Weighted Mean Pooling
        # sum(alpha * embeddings) along atom dimension
        weighted_mean = torch.sum(
            alpha * atomic_embeddings, dim=1
        )  # (Batch, LatentDim)

        # 2. Global Max Pooling
        # Mask padding for max pooling (set to large negative number)
        masked_embeddings = atomic_embeddings.masked_fill(~mask_expanded, -1e9)
        max_pool, _ = torch.max(masked_embeddings, dim=1)  # (Batch, LatentDim)

        # Concatenate
        return torch.cat([weighted_mean, max_pool], dim=1)


class CADSTFModel(nn.Module):
    """
    Context-Aware Deep Sets with Topological Fingerprinting.
    """

    def __init__(self):
        super().__init__()
        self.global_encoder = GlobalContextEncoder()
        self.atomic_encoder = ContextAwareAtomicEncoder()
        self.aggregator = AttentionAggregator()

        # Prediction Head
        # Input is LatentDim * 2 (from concat of mean and max pooling)
        self.head = nn.Sequential(
            nn.Linear(LATENT_DIM * 2, HEAD_HIDDEN_DIM),
            nn.BatchNorm1d(HEAD_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HEAD_HIDDEN_DIM, HEAD_HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(HEAD_HIDDEN_DIM // 2, 2),  # Output: Formation E, Bandgap E
        )

    def forward(self, global_features, atomic_features, mask):
        """
        Args:
            global_features: (Batch, NumGlobalFeatures)
            atomic_features: (Batch, MaxAtoms, BaseAtomicDim)
            mask: (Batch, MaxAtoms)
        """
        batch_size, max_atoms, _ = atomic_features.shape

        # 1. Encode Global Context
        context_vec = self.global_encoder(global_features)  # (Batch, ContextDim)

        # 2. Early Context Injection
        # Expand context vector to (Batch, MaxAtoms, ContextDim)
        context_expanded = context_vec.unsqueeze(1).expand(-1, max_atoms, -1)

        # Concatenate with atomic features
        # atomic_augmented: (Batch, MaxAtoms, BaseAtomicDim + ContextDim)
        atomic_augmented = torch.cat([atomic_features, context_expanded], dim=2)

        # 3. Atomic Encoding
        # Flatten for MLP: (Batch * MaxAtoms, InputDim)
        flat_input = atomic_augmented.view(-1, ATOMIC_INPUT_DIM)
        flat_embeddings = self.atomic_encoder(flat_input)

        # Reshape back: (Batch, MaxAtoms, LatentDim)
        atomic_embeddings = flat_embeddings.view(batch_size, max_atoms, LATENT_DIM)

        # 4. Aggregation
        graph_embedding = self.aggregator(atomic_embeddings, mask)

        # 5. Prediction
        output = self.head(graph_embedding)

        return output
