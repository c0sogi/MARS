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


class ZoneoutBiGRUBlock(nn.Module):
    """
    A Bidirectional GRU block with Zoneout regularization.
    Unrolls the sequence to apply stochastic state updates during training.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        zoneout_prob: float = Config.ZONEOUT_PROB,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.zoneout_prob = zoneout_prob

        # Forward and Backward Cells
        self.fwd_cell = nn.GRUCell(input_size, hidden_size)
        self.bwd_cell = nn.GRUCell(input_size, hidden_size)

    def forward(
        self, x: torch.Tensor, h_0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input sequence of shape (Batch, Seq_Len, Input_Size).
            h_0 (torch.Tensor, optional): Initial hidden state (Batch, Hidden_Size * 2).

        Returns:
            torch.Tensor: Output sequence of shape (Batch, Seq_Len, Hidden_Size * 2).
        """
        batch_size, seq_len, _ = x.size()

        # Initialize hidden states
        if h_0 is None:
            h_fwd = torch.zeros(
                batch_size, self.hidden_size, device=x.device, dtype=x.dtype
            )
            h_bwd = torch.zeros(
                batch_size, self.hidden_size, device=x.device, dtype=x.dtype
            )
        else:
            # Assuming h_0 is concatenated [h_fwd, h_bwd]
            h_fwd = h_0[:, : self.hidden_size]
            h_bwd = h_0[:, self.hidden_size :]

        outputs_fwd = []
        outputs_bwd = []

        # Pre-generate masks for efficiency if training
        # We need masks for every step: (Seq_Len, Batch, Hidden_Size)
        use_zoneout = self.training and self.zoneout_prob > 0.0

        # Forward Pass
        for t in range(seq_len):
            x_t = x[:, t, :]
            h_next = self.fwd_cell(x_t, h_fwd)

            if use_zoneout:
                # Zoneout: h_t = d_t * h_next + (1 - d_t) * h_prev
                # d_t ~ Bernoulli(1 - prob) (1 means keep new, 0 means keep old)
                # We sample a mask where 1 = update, 0 = keep old
                mask = torch.bernoulli(torch.full_like(h_next, 1.0 - self.zoneout_prob))
                h_fwd = mask * h_next + (1 - mask) * h_fwd
            elif self.zoneout_prob > 0.0:
                # Evaluation: Expectation
                # h_t = (1 - p) * h_next + p * h_prev
                h_fwd = (1 - self.zoneout_prob) * h_next + self.zoneout_prob * h_fwd
            else:
                # Standard GRU update
                h_fwd = h_next

            outputs_fwd.append(h_fwd)

        # Backward Pass
        for t in range(seq_len - 1, -1, -1):
            x_t = x[:, t, :]
            h_next = self.bwd_cell(x_t, h_bwd)

            if use_zoneout:
                mask = torch.bernoulli(torch.full_like(h_next, 1.0 - self.zoneout_prob))
                h_bwd = mask * h_next + (1 - mask) * h_bwd
            elif self.zoneout_prob > 0.0:
                h_bwd = (1 - self.zoneout_prob) * h_next + self.zoneout_prob * h_bwd
            else:
                h_bwd = h_next

            outputs_bwd.insert(0, h_bwd)

        # Stack and Concatenate
        # outputs_fwd: List of (B, H) -> (B, L, H)
        out_fwd = torch.stack(outputs_fwd, dim=1)
        out_bwd = torch.stack(outputs_bwd, dim=1)

        # (B, L, 2*H)
        output = torch.cat([out_fwd, out_bwd], dim=2)

        return output


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
