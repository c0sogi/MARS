import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple
from library.config import Config


class SinusoidalPairingEmbedding(nn.Module):
    """
    Encodes signed scalar pairing distances into fixed high-dimensional sinusoidal vectors.
    Uses sine and cosine functions of varying frequencies, similar to Transformer positional encodings,
    but applied to the pairing distance values.
    """

    def __init__(self, embed_dim: int = Config.EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim

        # Pre-compute the frequency divisors
        # div_term = 10000^(2i/d_model)
        # We compute this once and register as buffer
        half_dim = embed_dim // 2
        div_term = torch.exp(
            torch.arange(0, half_dim, 2).float() * -(math.log(10000.0) / half_dim)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, pair_dists: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair_dists (torch.Tensor): Signed distances of shape (Batch, Seq_Len).
                                       Values are floats (e.g., -5.0, 12.0, 0.0).
        Returns:
            torch.Tensor: Embeddings of shape (Batch, Seq_Len, Embed_Dim).
        """
        # pair_dists: (B, L) -> (B, L, 1)
        x = pair_dists.unsqueeze(-1)

        # div_term: (D/2,)
        # x * div_term -> (B, L, D/2)
        # Note: We only compute for half the dimension because sin/cos pairs take 2 slots
        scaled_dists = x * self.div_term

        # Create the encoding
        # We need to interleave sin and cos or concat them.
        # Standard implementation often concats sin(x) and cos(x).
        pe_sin = torch.sin(scaled_dists)
        pe_cos = torch.cos(scaled_dists)

        # Concatenate along the last dimension -> (B, L, D) (approx)
        # If embed_dim is odd, this logic needs adjustment, but typically it's even (128).
        pe = torch.cat([pe_sin, pe_cos], dim=-1)

        # If embed_dim was not perfectly divisible by the logic above, pad or trim.
        # With standard even dims (e.g. 128), this matches perfectly.
        if pe.shape[-1] != self.embed_dim:
            # Fallback for odd dimensions or mismatches (though unlikely with standard configs)
            pe = F.pad(pe, (0, self.embed_dim - pe.shape[-1]))

        return pe


class BiGRUBlock(nn.Module):
    """
    Standard Bidirectional GRU Block with Dropout.
    Uses cuDNN implementation for speed.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = Config.DROPOUT,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Seq, Input_Size)
        # out: (Batch, Seq, Hidden_Size * 2)
        out, _ = self.gru(x)
        return self.dropout(out)


class ScalarMixture(nn.Module):
    """
    Computes a learnable weighted sum of a list of input tensors.
    Used to aggregate representations from different layers.
    """

    def __init__(self, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        # Initialize weights uniformly
        self.weights = nn.Parameter(torch.ones(num_layers) / num_layers)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            inputs (List[torch.Tensor]): List of N tensors, each of shape (Batch, ...).
                                         All tensors must have the same shape.
        Returns:
            torch.Tensor: Weighted sum of inputs.
        """
        if len(inputs) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} inputs, got {len(inputs)}")

        # Stack inputs: (Batch, ..., Num_Layers)
        # It is more efficient to stack along a new dimension
        # Let's stack along the last dimension for broadcasting
        # Shape: (Batch, Seq, Dim, Num_Layers)
        stacked = torch.stack(inputs, dim=-1)

        # Normalize weights using Softmax to ensure stability and summing to 1
        norm_weights = F.softmax(self.weights, dim=0)

        # Weighted sum
        # (Batch, Seq, Dim, Num_Layers) * (Num_Layers) -> Sum over last dim
        weighted_sum = torch.sum(stacked * norm_weights, dim=-1)

        return weighted_sum
