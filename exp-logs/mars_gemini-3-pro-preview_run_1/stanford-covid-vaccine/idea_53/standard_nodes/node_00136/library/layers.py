import torch
import torch.nn as nn
import math
from library.config import Config


# Removed LayerNormLSTMCell and LayerNormBiLSTM in favor of Pre-LayerNorm GRU architecture
# Cite solution_lesson_node_00135


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Implements Fixed Sinusoidal Encodings for pairing distances.
    Preserves the sign to distinguish upstream/downstream dependencies.
    """

    def __init__(self, embedding_dim):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

        # Precompute the division term for sinusoidal calculation
        # div_term = 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * -(math.log(10000.0) / embedding_dim)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, positions):
        """
        Args:
            positions: Tensor of signed distances/positions.
                       Shape: (Batch, Seq_Len) or (Batch, Seq_Len, 1)

        Returns:
            embeddings: (Batch, Seq_Len, Embedding_Dim)
        """
        # Ensure input is float and has correct shape
        if positions.dim() == 3 and positions.size(-1) == 1:
            positions = positions.squeeze(-1)

        positions = positions.float()

        # Create output tensor
        # Shape: (Batch, Seq_Len, Embedding_Dim)
        pe = torch.zeros(
            positions.size(0),
            positions.size(1),
            self.embedding_dim,
            device=positions.device,
        )

        # Calculate sine and cosine
        # We use broadcasting: positions (B, S, 1) * div_term (1, 1, D/2)
        scaled_pos = positions.unsqueeze(-1) * self.div_term

        pe[..., 0::2] = torch.sin(scaled_pos)
        pe[..., 1::2] = torch.cos(scaled_pos)

        return pe
