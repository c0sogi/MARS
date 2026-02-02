import torch
import torch.nn as nn
import math
from typing import List
from library.config import Config


class SinusoidalSignedPositionalEmbedding(nn.Module):
    """
    Sinusoidal encoding for signed integers (pairing distances).
    Distinguishes between positive (downstream) and negative (upstream) distances
    by applying standard sinusoidal functions to the signed values.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Pre-compute the frequency constants
        # Frequencies are 1 / (10000 ** (2i / dim))
        # We compute for half the dimension since we concat sin and cos
        half_dim = dim // 2
        div_term = torch.exp(
            torch.arange(0, half_dim).float() * -(math.log(10000.0) / half_dim)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, SeqLen) containing signed integer distances.
        Returns:
            Tensor of shape (Batch, SeqLen, Dim)
        """
        # x is (Batch, SeqLen) -> unsqueeze to (Batch, SeqLen, 1)
        x_expanded = x.unsqueeze(-1).float()

        # div_term is (Dim/2)
        # Argument for trig functions: x * div_term -> (Batch, SeqLen, Dim/2)
        pe_arg = x_expanded * self.div_term

        # Compute sin and cos
        # sin(-x) = -sin(x), cos(-x) = cos(x)
        # The combination preserves the sign information.
        pe_sin = torch.sin(pe_arg)
        pe_cos = torch.cos(pe_arg)

        # Concatenate to get full dimension
        # Shape: (Batch, SeqLen, Dim)
        pe = torch.cat([pe_sin, pe_cos], dim=-1)

        return pe


class ResidualBiGRUBlock(nn.Module):
    """
    Wide-Stream Residual Block: Pre-LayerNorm -> BiGRU -> Dropout -> Residual.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        # BiGRU: Input is hidden_dim, Output is hidden_dim (2 * hidden_dim//2)
        self.bigru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (Batch, SeqLen, HiddenDim)
        Returns:
            Output tensor (Batch, SeqLen, HiddenDim)
        """
        residual = x

        # Pre-LayerNorm
        out = self.layer_norm(x)

        # BiGRU
        out, _ = self.bigru(out)

        # Dropout
        out = self.dropout(out)

        # Residual Connection
        return residual + out


class ScalarMixture(nn.Module):
    """
    Computes a learnable weighted sum of a list of tensors.
    """

    def __init__(self, num_layers: int):
        super().__init__()
        # Weights for [Stem, Block1, ..., BlockN]
        # Initialized to zeros -> uniform probability after softmax
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            tensors: List of tensors, all same shape (Batch, SeqLen, HiddenDim).
        Returns:
            Weighted sum tensor.
        """
        # Stack tensors: (NumLayers, Batch, SeqLen, HiddenDim)
        stacked = torch.stack(tensors, dim=0)

        # Softmax weights to ensure they sum to 1
        norm_weights = torch.softmax(self.weights, dim=0)

        # Weighted sum using broadcasting
        # weights: (NumLayers) -> (NumLayers, 1, 1, 1)
        weighted_sum = torch.sum(stacked * norm_weights.view(-1, 1, 1, 1), dim=0)

        return weighted_sum


class RNAModel(nn.Module):
    """
    Heterogeneous-Embedding Wide-Stream Residual BiGRU Model.
    """

    def __init__(self, config: Config):
        super().__init__()

        # 1. Embeddings
        # Atomic Sequence Identity
        self.seq_embedding = nn.Embedding(config.vocab_size, config.emb_dim_seq)
        # Predicted Loop Type
        self.loop_embedding = nn.Embedding(config.loop_types_size, config.emb_dim_loop)
        # Signed Sinusoidal Pairing Distance
        self.dist_embedding = SinusoidalSignedPositionalEmbedding(config.emb_dim_dist)

        # Total embedding dimension
        total_emb_dim = config.emb_dim_seq + config.emb_dim_loop + config.emb_dim_dist

        # 2. Stem
        # Projects concatenated embeddings to hidden_dim via BiGRU
        # This serves as the initial temporal contextualization
        self.stem = nn.GRU(
            input_size=total_emb_dim,
            hidden_size=config.hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )

        # 3. Backbone (Residual Blocks)
        self.blocks = nn.ModuleList(
            [
                ResidualBiGRUBlock(config.hidden_dim, config.dropout)
                for _ in range(config.num_layers)
            ]
        )

        # 4. Scalar Mixture
        # Inputs to mixture: Stem output + N Block outputs
        # We explicitly exclude raw embeddings to reduce noise.
        self.mixture = ScalarMixture(num_layers=1 + config.num_layers)

        # 5. Output Head
        # Shared projection to 3 targets
        self.head = nn.Linear(config.hidden_dim, len(config.target_cols))

    def forward(
        self, seqs: torch.Tensor, loops: torch.Tensor, dists: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            seqs: (Batch, SeqLen) - Sequence tokens
            loops: (Batch, SeqLen) - Loop type tokens
            dists: (Batch, SeqLen) - Signed distance integers
        Returns:
            (Batch, SeqLen, NumTargets)
        """
        # 1. Embeddings
        emb_seq = self.seq_embedding(seqs)  # (B, L, 128)
        emb_loop = self.loop_embedding(loops)  # (B, L, 64)
        emb_dist = self.dist_embedding(dists)  # (B, L, 64)

        # Concatenate heterogeneous embeddings
        x = torch.cat([emb_seq, emb_loop, emb_dist], dim=-1)  # (B, L, 256)

        # 2. Stem
        # Project to Wide-Stream dimension (512)
        x, _ = self.stem(x)  # (B, L, 512)

        # Collect outputs for scalar mixture
        layer_outputs = [x]

        # 3. Backbone
        for block in self.blocks:
            x = block(x)
            layer_outputs.append(x)

        # 4. Aggregation
        # Weighted sum of all layers
        x_agg = self.mixture(layer_outputs)  # (B, L, 512)

        # 5. Head
        logits = self.head(x_agg)  # (B, L, 3)

        return logits
