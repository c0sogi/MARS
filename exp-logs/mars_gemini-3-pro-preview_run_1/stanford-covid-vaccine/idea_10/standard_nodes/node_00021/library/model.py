import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config, SinusoidalPositionalEmbedding


class PreLNResidualBiGRU(nn.Module):
    """
    A Pre-LayerNorm Residual Bidirectional GRU block.
    Implements the structure: x = x + Dropout(GRU(LayerNorm(x)))
    This improves gradient flow and training stability for deeper networks.
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x, _ = self.gru(x)
        x = self.dropout(x)
        return x + residual


class RNAResidualBiGRU(nn.Module):
    """
    Distance-Aware Residual BiGRU.
    Reverts to the high-performing architecture from Lesson 17.

    Inputs: Sequence, Loop Type, Sinusoidal Pair Distance.
    Backbone: Deep Pre-LayerNorm Residual BiGRU.
    Head: Regression only (No MLM).
    """

    def __init__(self, config=Config()):
        super().__init__()
        self.config = config

        # 1. Feature Embeddings
        self.seq_embed = nn.Embedding(config.VOCAB_SIZE, config.EMBED_DIM)
        self.loop_embed = nn.Embedding(config.LOOP_VOCAB_SIZE, config.EMBED_DIM)
        self.dist_embed = SinusoidalPositionalEmbedding(config.EMBED_DIM)

        # 2. Input Projection
        # Concatenating 3 features: Seq + Loop + Dist
        input_dim = config.EMBED_DIM * 3
        self.input_proj = nn.Linear(input_dim, config.HIDDEN_DIM)
        self.dropout = nn.Dropout(config.DROPOUT)

        # 3. Backbone (Pre-LayerNorm Residual BiGRU - Cite solution_lesson_node_00017)
        self.layers = nn.ModuleList(
            [
                PreLNResidualBiGRU(config.HIDDEN_DIM, config.DROPOUT)
                for _ in range(config.NUM_LAYERS)
            ]
        )

        # 4. Output Head (Regression Only)
        self.reg_head = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM // 2, 3),
        )

    def forward(self, seq, loop, pair_dist):
        """
        Args:
            seq: (B, L) Sequence token indices.
            loop: (B, L) Loop type token indices.
            pair_dist: (B, L) Geometric distances between pairs.

        Returns:
            reg_out: (B, L, 3) Regression predictions.
        """
        # 1. Embeddings
        x_seq = self.seq_embed(seq)  # (B, L, D)
        x_loop = self.loop_embed(loop)  # (B, L, D)
        x_dist = self.dist_embed(pair_dist)  # (B, L, D)

        # 2. Combine Features & Project
        x = torch.cat([x_seq, x_loop, x_dist], dim=-1)
        x = self.input_proj(x)
        x = self.dropout(x)

        # 3. Backbone Processing
        for layer in self.layers:
            x = layer(x)

        # 4. Head
        reg_out = self.reg_head(x)

        return reg_out
