import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import Config


class SinusoidalPositionalEncoding(nn.Module):
    """
    Implements fixed sinusoidal positional encodings for signed scalar values.
    Used for encoding pairing distances (j - i).
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Precompute the division term for the geometric progression of frequencies
        # div_term = 10000^(-2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len) containing signed distances (float).
        Returns:
            Tensor of shape (Batch, Seq_Len, d_model)
        """
        # Unsqueeze to broadcast: (B, L, 1)
        x_unsqueezed = x.unsqueeze(-1)

        # Compute phase: (B, L, d_model/2)
        phase = x_unsqueezed * self.div_term

        # Initialize output tensor
        pe = torch.zeros(x.size(0), x.size(1), self.d_model, device=x.device)

        # Apply sin to even indices and cos to odd indices
        pe[:, :, 0::2] = torch.sin(phase)
        pe[:, :, 1::2] = torch.cos(phase)

        return pe


class ResidualBiGRUBlock(nn.Module):
    """
    A Wide-Stream Residual Block using Pre-LayerNorm, BiGRU, and Inter-Layer Dropout.
    Structure: x = x + Dropout(BiGRU(LN(x)))
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        # BiGRU projects hidden_dim -> hidden_dim (hidden_dim//2 * 2)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        # Pre-LayerNorm
        out = self.ln(x)

        # BiGRU
        out, _ = self.gru(out)

        # Inter-Layer Dropout
        out = self.dropout(out)

        # Residual Connection
        return residual + out


class InputInclusiveScalarMixture(nn.Module):
    """
    Aggregates multiple tensors using learnable scalar weights (Softmax normalized).
    """

    def __init__(self, num_sources):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_sources))

    def forward(self, tensors):
        """
        Args:
            tensors: List of tensors, each of shape (Batch, Seq_Len, Hidden_Dim)
        Returns:
            Weighted sum tensor of shape (Batch, Seq_Len, Hidden_Dim)
        """
        # Stack: (Batch, Seq_Len, Hidden_Dim, Num_Sources)
        stacked = torch.stack(tensors, dim=-1)

        # Calculate normalized weights
        norm_weights = F.softmax(self.weights, dim=0)

        # Weighted sum
        weighted_sum = torch.sum(stacked * norm_weights, dim=-1)

        return weighted_sum


class RNAModel(nn.Module):
    """
    Input-Inclusive Wide-Stream Residual BiGRU Model.
    """

    def __init__(self, config=Config):
        super().__init__()

        self.hidden_dim = config.HIDDEN_DIM
        self.embedding_dim = config.EMBEDDING_DIM

        # ----------------------------------------------------------------------
        # 1. Embeddings
        # ----------------------------------------------------------------------
        # Atomic Sequence Embedding (A, G, C, U)
        self.seq_embedding = nn.Embedding(4, self.embedding_dim)

        # Predicted Loop Type Embedding (S, M, I, B, H, E, X)
        self.loop_embedding = nn.Embedding(7, self.embedding_dim)

        # Signed Sinusoidal Pairing Distance Embedding
        self.dist_embedding = SinusoidalPositionalEncoding(self.embedding_dim)

        # Total Input Dimension (concatenated)
        self.input_dim = self.embedding_dim * 3

        # ----------------------------------------------------------------------
        # 2. Input Projection (for Input-Inclusive Mixture)
        # ----------------------------------------------------------------------
        # Projects raw embeddings to hidden_dim to be included in the mixture
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)

        # ----------------------------------------------------------------------
        # 3. Recurrent Stem
        # ----------------------------------------------------------------------
        # Initial projection to the residual stream width
        self.stem = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim // 2,
            bidirectional=True,
            batch_first=True,
        )

        # ----------------------------------------------------------------------
        # 4. Backbone (Wide-Stream Residual Blocks)
        # ----------------------------------------------------------------------
        self.layers = nn.ModuleList(
            [
                ResidualBiGRUBlock(self.hidden_dim, dropout=config.DROPOUT)
                for _ in range(config.N_LAYERS)
            ]
        )

        # ----------------------------------------------------------------------
        # 5. Scalar Mixture
        # ----------------------------------------------------------------------
        # Sources: Input_Proj (1) + Stem (1) + Layers (N_LAYERS)
        num_sources = 2 + config.N_LAYERS
        self.mixture = InputInclusiveScalarMixture(num_sources)

        # ----------------------------------------------------------------------
        # 6. Output Head
        # ----------------------------------------------------------------------
        # Shared linear projection to 3 targets
        self.head = nn.Linear(self.hidden_dim, 3)

    def forward(self, seq, loop, dist):
        # 1. Embed Inputs
        emb_seq = self.seq_embedding(seq)  # (B, L, 128)
        emb_loop = self.loop_embedding(loop)  # (B, L, 128)
        emb_dist = self.dist_embedding(dist)  # (B, L, 128)

        # Concatenate
        x_in = torch.cat([emb_seq, emb_loop, emb_dist], dim=-1)  # (B, L, 384)

        # Collect tensors for mixture
        mixture_inputs = []

        # Source A: Projected Input (Safe Shortcut)
        x_proj = self.input_proj(x_in)
        mixture_inputs.append(x_proj)

        # Source B: Stem Output
        x_stem, _ = self.stem(x_in)
        mixture_inputs.append(x_stem)

        # Source C: Layer Outputs
        x = x_stem
        for layer in self.layers:
            x = layer(x)
            mixture_inputs.append(x)

        # 2. Aggregate
        x_agg = self.mixture(mixture_inputs)

        # 3. Predict
        logits = self.head(x_agg)

        return logits
