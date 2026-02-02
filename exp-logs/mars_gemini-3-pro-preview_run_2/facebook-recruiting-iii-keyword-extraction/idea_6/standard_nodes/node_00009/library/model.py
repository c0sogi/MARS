import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import VOCAB_SIZE_WIDE, VOCAB_SIZE_DEEP, EMBED_DIM, NUM_TAGS


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.
    Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    This loss dynamically scales the Cross Entropy Loss based on the confidence
    of the prediction, down-weighting easy examples and focusing on hard ones.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Probabilities (batch_size, num_classes)
            targets: Binary labels (batch_size, num_classes)
        """
        # Clamp probabilities to avoid log(0) and log(1)
        p = torch.clamp(inputs, min=1e-7, max=1.0 - 1e-7)

        # Compute loss terms for positive (target=1) and negative (target=0) cases
        # For targets == 1: -alpha * (1-p)^gamma * log(p)
        # For targets == 0: -(1-alpha) * p^gamma * log(1-p)

        loss_pos = -self.alpha * torch.pow(1.0 - p, self.gamma) * torch.log(p)
        loss_neg = -(1.0 - self.alpha) * torch.pow(p, self.gamma) * torch.log(1.0 - p)

        # Combine losses
        loss = targets * loss_pos + (1.0 - targets) * loss_neg

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class WideAndDeepModel(nn.Module):
    """
    Wide and Deep Model for Tag Prediction.

    Architecture:
    1. Wide Component: A sparse linear layer that takes TF-IDF vectors as input.
       Implemented efficiently using nn.EmbeddingBag with mode='sum' and per-sample weights.
       This captures direct keyword-to-tag mappings.

    2. Deep Component: A dense neural network that processes token sequences.
       Embedding -> Global Average Pooling -> Dense -> ReLU -> Dense.
       This captures semantic similarities and latent topics.

    3. Fusion: The logits from both components are summed and passed through a Sigmoid activation.
    """

    def __init__(self):
        super(WideAndDeepModel, self).__init__()

        # =====================================================================
        # Wide Component
        # =====================================================================
        # We use EmbeddingBag to simulate a Linear layer on sparse inputs (W * x).
        # num_embeddings = Vocabulary size of TF-IDF
        # embedding_dim = Number of output tags
        # mode = 'sum' combined with per_sample_weights implements the dot product logic.
        self.wide = nn.EmbeddingBag(
            num_embeddings=VOCAB_SIZE_WIDE,
            embedding_dim=NUM_TAGS,
            mode="sum",
            sparse=False,  # Use dense gradients to ensure compatibility with standard Adam
        )

        # Explicit bias term for the Wide component
        self.wide_bias = nn.Parameter(torch.zeros(NUM_TAGS))

        # Initialize Wide weights
        nn.init.xavier_uniform_(self.wide.weight)

        # =====================================================================
        # Deep Component
        # =====================================================================
        self.deep_embed = nn.Embedding(
            num_embeddings=VOCAB_SIZE_DEEP, embedding_dim=EMBED_DIM, padding_idx=0
        )

        # Hidden Dense Layers
        # We expand slightly to allow for interaction before projecting to tags
        hidden_dim = EMBED_DIM * 2

        self.deep_fc1 = nn.Linear(EMBED_DIM, hidden_dim)
        self.deep_relu = nn.ReLU()
        self.deep_dropout = nn.Dropout(0.2)
        self.deep_fc2 = nn.Linear(hidden_dim, NUM_TAGS)

        # Initialize Deep weights
        nn.init.xavier_uniform_(self.deep_embed.weight)
        nn.init.kaiming_normal_(
            self.deep_fc1.weight, mode="fan_in", nonlinearity="relu"
        )
        nn.init.xavier_uniform_(self.deep_fc2.weight)

    def forward(self, deep_seq, wide_indices, wide_values, wide_offsets):
        """
        Forward pass of the Wide and Deep model.

        Args:
            deep_seq: (batch_size, max_len) - Token indices for the Deep component.
            wide_indices: (total_nonzero) - Flattened feature indices for the Wide component.
            wide_values: (total_nonzero) - Flattened TF-IDF weights for the Wide component.
            wide_offsets: (batch_size) - Starting index in wide_indices for each sample.

        Returns:
            probs: (batch_size, num_tags) - Predicted probabilities in [0, 1].
        """
        # --- Deep Path ---
        # Embed: (batch, max_len) -> (batch, max_len, embed_dim)
        deep_out = self.deep_embed(deep_seq)

        # Global Average Pooling: (batch, max_len, embed_dim) -> (batch, embed_dim)
        deep_out = torch.mean(deep_out, dim=1)

        # Dense Layers
        deep_out = self.deep_fc1(deep_out)
        deep_out = self.deep_relu(deep_out)
        deep_out = self.deep_dropout(deep_out)
        deep_out = self.deep_fc2(deep_out)

        # --- Wide Path ---
        # EmbeddingBag computes the weighted sum of embeddings based on indices and values.
        # This is mathematically equivalent to W * x_sparse.
        wide_out = self.wide(
            wide_indices, offsets=wide_offsets, per_sample_weights=wide_values
        )
        wide_out = wide_out + self.wide_bias

        # --- Fusion ---
        # Sum logits from both paths
        logits = wide_out + deep_out

        # Apply Sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        return probs
