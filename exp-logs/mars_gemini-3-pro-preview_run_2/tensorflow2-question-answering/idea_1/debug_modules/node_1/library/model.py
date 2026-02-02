import torch
import torch.nn as nn
import numpy as np
from library.config import Config


class BoERanker(nn.Module):
    """
    Neural Bag-of-Embeddings Ranker.
    Encodes Question and Candidate text into fixed vectors using Global Average Pooling
    on pre-trained embeddings, computes interaction features, and scores relevance via MLP.
    """

    def __init__(self, embedding_matrix):
        """
        Args:
            embedding_matrix (np.array): Pre-trained embedding matrix of shape (vocab_size, embedding_dim).
        """
        super(BoERanker, self).__init__()

        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding_dim = embed_dim

        # 1. Embedding Layer
        # Load pre-trained weights
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,  # Config.PAD_TOKEN index is 0
        )
        # Initialize with provided matrix
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))

        # Freeze or Fine-tune
        if Config.FREEZE_EMBEDDINGS:
            self.embedding.weight.requires_grad = False

        # 2. Classifier Head (MLP)
        # Input features: [q_vec, c_vec, q*c, |q-c|] -> 4 * embed_dim
        input_dim = 4 * embed_dim
        layers = []
        prev_dim = input_dim

        for hidden_dim in Config.HIDDEN_DIMS:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(Config.DROPOUT_RATE))
            prev_dim = hidden_dim

        # Final scoring layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.classifier = nn.Sequential(*layers)

    def encode(self, seqs):
        """
        Encodes a batch of sequences into fixed vectors using Global Average Pooling.
        Handles padding masking.

        Args:
            seqs (torch.Tensor): Batch of token indices (batch_size, seq_len).

        Returns:
            torch.Tensor: Pooled vectors (batch_size, embed_dim).
        """
        # Get embeddings: (batch_size, seq_len, embed_dim)
        embedded = self.embedding(seqs)

        # Create mask for non-padding tokens (batch_size, seq_len, 1)
        # padding_idx is 0
        mask = (seqs != 0).unsqueeze(-1).float()

        # Sum embeddings along sequence dimension
        summed = torch.sum(embedded * mask, dim=1)

        # Count non-padding tokens
        counts = torch.sum(mask, dim=1)

        # Avoid division by zero
        counts = torch.clamp(counts, min=1.0)

        # Average
        averaged = summed / counts
        return averaged

    def forward(self, q_seqs, c_seqs):
        """
        Forward pass for ranking.

        Args:
            q_seqs (torch.Tensor): Question sequences (batch_size, seq_len).
            c_seqs (torch.Tensor): Candidate sequences (batch_size, seq_len).

        Returns:
            torch.Tensor: Probability scores (batch_size,).
        """
        # 1. Encode Question and Candidate
        q_vec = self.encode(q_seqs)  # (batch_size, embed_dim)
        c_vec = self.encode(c_seqs)  # (batch_size, embed_dim)

        # 2. Interaction Features
        # Element-wise product
        prod_vec = q_vec * c_vec
        # Absolute difference
        diff_vec = torch.abs(q_vec - c_vec)

        # 3. Concatenate Features
        # Shape: (batch_size, 4 * embed_dim)
        combined = torch.cat([q_vec, c_vec, prod_vec, diff_vec], dim=1)

        # 4. Score
        # Shape: (batch_size, 1) -> (batch_size,)
        scores = self.classifier(combined).squeeze(-1)

        return scores
