import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import Config


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed Sinusoidal Positional Encoding for signed distances.
    Cite Lesson 00057: Fixed inductive biases outperform learnable ones in low-data regimes.
    Cite Lesson 00042: Prefer Sinusoidal over RBF for 1D sequence distances.
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Pre-compute division term
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Signed distances (B, L).
        """
        # (B, L, 1)
        x = x.unsqueeze(-1)
        # (1, 1, D/2)
        div_term = self.div_term.view(1, 1, -1)

        # Compute phase: (B, L, D/2)
        phase = x * div_term

        # Interleave sin and cos
        pe = torch.zeros(x.size(0), x.size(1), self.d_model, device=x.device)
        pe[:, :, 0::2] = torch.sin(phase)
        pe[:, :, 1::2] = torch.cos(phase)

        return pe


class ResidualBiGRUBlock(nn.Module):
    """
    Pre-LayerNorm Wide-Stream Residual BiGRU Block.
    Cite Lesson 00017: Pre-LayerNorm facilitates scaling.
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()

        self.norm = nn.LayerNorm(hidden_dim)

        # Wide-stream BiGRU
        self.bigru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = x

        # 1. Pre-Norm
        x = self.norm(x)

        # 2. Recurrent Processing
        x, _ = self.bigru(x)

        # 3. Dropout
        x = self.dropout(x)

        # 4. Residual Connection
        out = identity + x
        return out


class ScalarMixtureAggregator(nn.Module):
    """
    Aggregates outputs from multiple layers using a learnable scalar mixture.
    """

    def __init__(self, num_layers):
        super().__init__()
        # Learnable weights for each layer
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, layer_outputs):
        """
        Args:
            layer_outputs (list): List of tensors [(B, L, D), ...].
        Returns:
            torch.Tensor: Weighted sum (B, L, D).
        """
        # Stack: (B, L, D, num_layers)
        stacked = torch.stack(layer_outputs, dim=-1)

        # Softmax weights to ensure they sum to 1
        norm_weights = F.softmax(self.weights, dim=0)

        # Weighted sum
        weighted_sum = torch.sum(stacked * norm_weights, dim=-1)

        return weighted_sum


class RNAModel(nn.Module):
    """
    Wide-Stream BiGRU with Fixed Geometric Bias and Scalar Aggregation.
    """

    def __init__(self, config=Config):
        super().__init__()

        self.hidden_dim = config.HIDDEN_DIM
        self.num_layers = config.NUM_LAYERS
        self.dropout_rate = config.DROPOUT

        # --- 1. Input Embeddings ---
        # Atomic Nucleotide Embedding
        self.seq_embedding = nn.Embedding(len(config.NUCLEOTIDE_MAP), 32)

        # Loop Embedding
        self.loop_embedding = nn.Embedding(len(config.LOOP_TYPE_MAP), 32)

        # Fixed Sinusoidal Positional Encoding
        # Cite Lesson 00057: Use deterministic embeddings for geometric properties
        self.dist_encoding = SinusoidalPositionalEncoding(d_model=64)
        dist_dim = 64

        # Total dimension entering the Stem
        input_dim = 32 + 32 + dist_dim  # 128

        # --- 2. Recurrent Stem ---
        # Projects concatenated embeddings to the residual stream width
        self.stem = nn.GRU(
            input_size=input_dim,
            hidden_size=self.hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.stem_dropout = nn.Dropout(self.dropout_rate)

        # --- 3. Backbone ---
        self.blocks = nn.ModuleList(
            [
                ResidualBiGRUBlock(
                    hidden_dim=self.hidden_dim,
                    dropout=self.dropout_rate,
                )
                for _ in range(self.num_layers)
            ]
        )

        # --- 4. Output Head ---
        # Aggregates Stem + N Blocks (Cite Lesson 00049)
        self.aggregator = ScalarMixtureAggregator(num_layers=1 + self.num_layers)

        # Final projection to the 3 scored targets
        num_targets = len(config.TARGET_COLS)
        self.head = nn.Linear(self.hidden_dim, num_targets)

    def forward(self, seq, loop, dist):
        """
        Args:
            seq (torch.Tensor): Sequence indices (B, L).
            loop (torch.Tensor): Loop type indices (B, L).
            dist (torch.Tensor): Signed pairing distances (B, L).
        """
        # Embeddings
        seq_emb = self.seq_embedding(seq)  # (B, L, 32)
        loop_emb = self.loop_embedding(loop)  # (B, L, 32)
        dist_emb = self.dist_encoding(dist)  # (B, L, 64)

        # Concatenate inputs (Cite Lesson 00057: Linear feature fusion)
        x = torch.cat([seq_emb, loop_emb, dist_emb], dim=-1)  # (B, L, 128)

        # Stem Processing
        x, _ = self.stem(x)  # (B, L, hidden_dim)
        x = self.stem_dropout(x)

        # Store outputs for aggregation
        layer_outputs = [x]

        # Pass through Residual Blocks
        for block in self.blocks:
            x = block(x)
            layer_outputs.append(x)

        # Aggregate outputs
        x_agg = self.aggregator(layer_outputs)

        # Project to targets
        logits = self.head(x_agg)  # (B, L, 3)

        return logits
