import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import Config


class DIPNRanker(nn.Module):
    """
    Direct Interaction Pooling Network (DIPN) for Long Answer Ranking.

    Architecture:
    1. Embedding Layer (Pre-trained)
    2. Word-to-Word Interaction Matrix (Dot Product)
    3. Direct Pooling:
       - Row-wise Max Pooling (Best match for each Q token)
       - Column-wise Mean Pooling (Average relevance for each P token)
    4. MLP Scoring Head
    """

    def __init__(self, embedding_matrix):
        """
        Args:
            embedding_matrix (numpy.ndarray): Pre-trained embedding matrix of shape (vocab_size, embedding_dim).
        """
        super(DIPNRanker, self).__init__()

        # Convert numpy matrix to tensor
        embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float32)
        num_embeddings, embedding_dim = embedding_tensor.shape

        # 1. Embedding Layer
        # We allow fine-tuning (freeze=False) to adapt embeddings to the specific QA task
        self.embedding = nn.Embedding.from_pretrained(
            embedding_tensor, freeze=False, padding_idx=0
        )

        # Calculate input dimension for the MLP
        # The pooling outputs a vector of size MAX_Q_LEN (row-wise max)
        # and a vector of size MAX_DOC_LEN (col-wise mean).
        # These are concatenated.
        mlp_input_dim = Config.MAX_Q_LEN + Config.MAX_DOC_LEN

        # 2. MLP Scoring Head
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, Config.RANKER_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.RANKER_DROPOUT),
            nn.Linear(Config.RANKER_HIDDEN_DIM, 1),
        )

    def forward(self, q_indices, p_indices):
        """
        Args:
            q_indices (torch.Tensor): Question token indices, shape (batch_size, max_q_len)
            p_indices (torch.Tensor): Paragraph token indices, shape (batch_size, max_doc_len)

        Returns:
            torch.Tensor: Relevance logits, shape (batch_size,)
        """
        # Create masks for padding (index 0)
        # Shape: (batch_size, seq_len)
        q_mask = (q_indices != 0).float()
        p_mask = (p_indices != 0).float()

        # Embeddings
        # Shape: (batch_size, seq_len, embedding_dim)
        q_embed = self.embedding(q_indices)
        p_embed = self.embedding(p_indices)

        # Compute Interaction Matrix (Dot Product)
        # (B, Q, E) @ (B, P, E)^T -> (B, Q, P)
        interaction_matrix = torch.bmm(q_embed, p_embed.transpose(1, 2))

        # ---------------------------------------------------------
        # Direct Pooling Operations
        # ---------------------------------------------------------

        # 1. Row-wise Max Pooling (over Paragraph dimension)
        # "How well is each question term matched?"
        # We need to mask out padding in the paragraph dimension so it doesn't affect max.
        # p_mask shape is (B, P). Unsqueeze to (B, 1, P) for broadcasting.
        # Set padding positions to a very small number (-1e9)
        p_mask_expanded = p_mask.unsqueeze(1)  # (B, 1, P)

        # Apply mask: where mask is 0, replace interaction with -inf
        masked_interaction_for_max = interaction_matrix.masked_fill(
            p_mask_expanded == 0, -1e9
        )

        # Max over P dimension (dim 2)
        # Result shape: (B, Q)
        row_max_pool, _ = torch.max(masked_interaction_for_max, dim=2)

        # 2. Column-wise Mean Pooling (over Question dimension)
        # "How relevant is this text segment overall?"
        # We need to average over the question dimension, ignoring padding in Q.
        # q_mask shape is (B, Q). Unsqueeze to (B, Q, 1).
        q_mask_expanded = q_mask.unsqueeze(2)  # (B, Q, 1)

        # Zero out interactions corresponding to padded question tokens
        masked_interaction_for_mean = interaction_matrix * q_mask_expanded

        # Sum over Q dimension (dim 1)
        sum_over_q = torch.sum(masked_interaction_for_mean, dim=1)  # (B, P)

        # Count valid tokens in Q for each batch element
        q_lengths = torch.sum(q_mask, dim=1, keepdim=True)  # (B, 1)

        # Avoid division by zero
        q_lengths = torch.clamp(q_lengths, min=1.0)

        # Compute mean
        # Result shape: (B, P)
        col_mean_pool = sum_over_q / q_lengths

        # ---------------------------------------------------------
        # Aggregation and Scoring
        # ---------------------------------------------------------

        # Concatenate the pooled vectors
        # Shape: (B, Q + P)
        # Note: Since the input tensors are padded to MAX_Q_LEN and MAX_DOC_LEN in the collate function,
        # the pooled vectors naturally have these fixed dimensions.
        pooled_features = torch.cat([row_max_pool, col_mean_pool], dim=1)

        # Pass through MLP
        logits = self.mlp(pooled_features)

        # Remove last dimension to get (B,)
        return logits.squeeze(-1)
