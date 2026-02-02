import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from library import config

# -----------------------------------------------------------------------------
# Reproducibility Setup
# -----------------------------------------------------------------------------
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
random.seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.SEED)


class HighwayLayer(nn.Module):
    """
    Implements a Highway Layer which allows information to flow unimpeded
    across layers using a gating mechanism.

    Formula: y = H(x, Wh) * T(x, Wt) + x * (1 - T(x, Wt))
    where H is the non-linear transform and T is the transform gate.
    """

    def __init__(self, input_dim, dropout_rate=None):
        """
        Args:
            input_dim (int): Dimension of the input tensor.
            dropout_rate (float, optional): Dropout probability. Defaults to config.DROPOUT_RATE.
        """
        super(HighwayLayer, self).__init__()
        if dropout_rate is None:
            dropout_rate = config.DROPOUT_RATE

        self.transform_gate = nn.Linear(input_dim, input_dim)
        self.nonlinear_transform = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, ..., input_dim)

        Returns:
            torch.Tensor: Output tensor of same shape as input.
        """
        # T(x) = sigmoid(W_t * x + b_t)
        transform_gate = torch.sigmoid(self.transform_gate(x))

        # H(x) = ReLU(W_h * x + b_h)
        nonlinear = torch.relu(self.nonlinear_transform(x))
        nonlinear = self.dropout(nonlinear)

        # y = H(x) * T(x) + x * (1 - T(x))
        output = nonlinear * transform_gate + x * (1 - transform_gate)
        return output


class CoAttention(nn.Module):
    """
    Computes Context-to-Query Attention.

    It calculates a similarity matrix between context and query, determines which
    query words are relevant to each context word, and fuses this information
    with the original context embeddings.
    """

    def __init__(self, input_dim):
        """
        Args:
            input_dim (int): Dimension of the input embeddings (D).
        """
        super(CoAttention, self).__init__()
        # Linear projection for bilinear similarity computation
        # We compute similarity S = (C * W) * Q^T
        self.W = nn.Linear(input_dim, input_dim, bias=False)

    def forward(self, query, context, query_mask=None):
        """
        Args:
            query (torch.Tensor): Query embeddings (Batch, Q_Len, Dim).
            context (torch.Tensor): Context embeddings (Batch, C_Len, Dim).
            query_mask (torch.Tensor, optional): Mask for query tokens (Batch, Q_Len).

        Returns:
            torch.Tensor: Fused representation (Batch, C_Len, 2 * Dim).
        """
        # Project context: (B, C_Len, D) -> (B, C_Len, D)
        proj_context = self.W(context)

        # Compute Similarity Matrix S: (B, C_Len, D) @ (B, D, Q_Len) -> (B, C_Len, Q_Len)
        # S_ij represents similarity between context word i and query word j
        similarity = torch.bmm(proj_context, query.transpose(1, 2))

        # Apply mask to query dimension if provided to ignore padding in query
        if query_mask is not None:
            # query_mask shape: (B, Q_Len) -> (B, 1, Q_Len)
            mask = query_mask.unsqueeze(1)
            # Set masked positions to a large negative number so softmax becomes 0
            similarity = similarity.masked_fill(mask == 0, -1e9)

        # Context-to-Query Attention weights
        # For each context word, we get a distribution over query words
        # Shape: (B, C_Len, Q_Len)
        c2q_weights = F.softmax(similarity, dim=-1)

        # Compute attended query representation for each context word
        # (B, C_Len, Q_Len) @ (B, Q_Len, D) -> (B, C_Len, D)
        attended_query = torch.bmm(c2q_weights, query)

        # Fuse: Concatenate original context with attended query
        # Shape: (B, C_Len, 2 * Dim)
        fused_context = torch.cat([context, attended_query], dim=-1)

        return fused_context


class KMaxPooling(nn.Module):
    """
    Implements K-Max Interaction Pooling for the Ranker.

    1. Computes interaction matrix between Query and Candidate.
    2. For each Query token, selects Top-K highest similarity scores from Candidate.
    3. Aggregates these scores via Sum-Pooling to produce a fixed-size vector.
    """

    def __init__(self, k=None):
        """
        Args:
            k (int, optional): The number of top interactions to pool.
                               Defaults to config.K_MAX.
        """
        super(KMaxPooling, self).__init__()
        self.k = k if k is not None else config.K_MAX

    def forward(self, query, candidate, query_mask=None, candidate_mask=None):
        """
        Args:
            query (torch.Tensor): Query embeddings (Batch, Q_Len, Dim).
            candidate (torch.Tensor): Candidate embeddings (Batch, C_Len, Dim).
            query_mask (torch.Tensor, optional): Mask for query tokens (Batch, Q_Len).
            candidate_mask (torch.Tensor, optional): Mask for candidate tokens (Batch, C_Len).

        Returns:
            torch.Tensor: Pooled interaction feature vector (Batch, K).
        """
        # Compute Interaction Matrix (Dot Product)
        # (B, Q_Len, D) @ (B, D, C_Len) -> (B, Q_Len, C_Len)
        interaction = torch.bmm(query, candidate.transpose(1, 2))

        # Apply candidate mask (columns)
        # We want to ignore interactions with padding tokens in the candidate
        if candidate_mask is not None:
            # (B, 1, C_Len)
            c_mask = candidate_mask.unsqueeze(1)
            # Mask with large negative value so they don't appear in top-k
            interaction = interaction.masked_fill(c_mask == 0, -1e9)

        # K-Max Pooling
        # For each query token (row), find top K scores across candidate (cols)
        c_len = interaction.size(2)
        curr_k = min(self.k, c_len)

        # topk returns (values, indices)
        # values shape: (B, Q_Len, K)
        topk_values, _ = torch.topk(interaction, k=curr_k, dim=2)

        # If actual candidate length is less than required K, pad the results
        if curr_k < self.k:
            pad_size = self.k - curr_k
            # Pad the last dimension with a value that won't affect the sum significantly
            # relative to high scores, or 0 if we assume ReLU-like behavior elsewhere.
            # However, for pure similarity, padding with 0 is safer for sum pooling
            # if we assume similarity is roughly centered or positive.
            # Given standard embeddings, dot products can be negative.
            # But since this is a feature vector for an MLP, 0 padding is standard convention.
            topk_values = F.pad(topk_values, (0, pad_size), value=0.0)

        # Apply query mask (rows)
        # We want to ignore rows corresponding to padding tokens in the query
        # so they don't contribute to the sum aggregation.
        if query_mask is not None:
            # (B, Q_Len, 1)
            q_mask = query_mask.unsqueeze(2)
            # Zero out the values for masked query tokens
            topk_values = topk_values * q_mask.float()

        # Aggregation: Sum-Pooling across query tokens
        # Sum across dimension 1 (Q_Len)
        # Result shape: (B, K)
        pooled_features = torch.sum(topk_values, dim=1)

        return pooled_features
