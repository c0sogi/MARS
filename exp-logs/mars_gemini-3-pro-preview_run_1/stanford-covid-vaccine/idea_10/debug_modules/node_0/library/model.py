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


class RNAMultiTaskBiGRU(nn.Module):
    """
    Multi-Task Enhanced Distance-Aware Residual BiGRU.

    Key Components:
    1. Inputs: Masked Sequence, Loop Type, Sinusoidal Pair Distance.
    2. Innovation: Paired-Base Identity Feature (Teleportation of semantic info).
    3. Backbone: Deep Pre-LayerNorm Residual BiGRU.
    4. Heads:
       - Regression: Predicts 3 scored targets (reactivity, deg_Mg_pH10, deg_Mg_50C).
       - Reconstruction: Predicts masked bases (A, G, C, U) for regularization.
    """

    def __init__(self, config=Config()):
        super().__init__()
        self.config = config

        # 1. Feature Embeddings
        self.seq_embed = nn.Embedding(config.VOCAB_SIZE, config.EMBED_DIM)
        self.loop_embed = nn.Embedding(config.LOOP_VOCAB_SIZE, config.EMBED_DIM)
        self.dist_embed = SinusoidalPositionalEmbedding(config.EMBED_DIM)

        # 2. Input Projection
        # Concatenating 4 features: Seq + Loop + Dist + PairedSeq
        input_dim = config.EMBED_DIM * 4
        self.input_proj = nn.Linear(input_dim, config.HIDDEN_DIM)
        self.dropout = nn.Dropout(config.DROPOUT)

        # 3. Backbone
        self.layers = nn.ModuleList(
            [
                PreLNResidualBiGRU(config.HIDDEN_DIM, config.DROPOUT)
                for _ in range(config.NUM_LAYERS)
            ]
        )

        # 4. Output Heads
        # Regression Head: Predicts the 3 scored columns
        self.reg_head = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM // 2, 3),
        )

        # Reconstruction Head: Predicts the 4 nucleotide bases
        self.recon_head = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM // 2, 4),
        )

    def forward(self, seq, loop, pair_idx, pair_dist):
        """
        Args:
            seq: (B, L) Sequence token indices.
            loop: (B, L) Loop type token indices.
            pair_idx: (B, L) Indices of paired bases (-1 for unpaired).
            pair_dist: (B, L) Geometric distances between pairs.

        Returns:
            reg_out: (B, L, 3) Regression predictions.
            recon_out: (B, L, 4) Reconstruction logits.
        """
        # 1. Basic Embeddings
        x_seq = self.seq_embed(seq)  # (B, L, D)
        x_loop = self.loop_embed(loop)  # (B, L, D)
        x_dist = self.dist_embed(pair_dist)  # (B, L, D)

        # 2. Paired-Base Identity Feature
        # "Teleport" the embedding of the paired base to the current position.
        # pair_idx contains the index of the pair. Unpaired is -1.

        # Handle unpaired indices (-1) by clamping to 0 temporarily
        safe_pair_idx = pair_idx.clone()
        unpaired_mask = safe_pair_idx == -1
        safe_pair_idx[unpaired_mask] = 0

        # Expand indices for gather: (B, L, D)
        idx_expanded = safe_pair_idx.unsqueeze(-1).expand(-1, -1, self.config.EMBED_DIM)

        # Gather embeddings from x_seq using the paired indices
        x_paired = torch.gather(x_seq, 1, idx_expanded)

        # Zero out features for unpaired positions
        mask_expanded = unpaired_mask.unsqueeze(-1).expand(
            -1, -1, self.config.EMBED_DIM
        )
        x_paired[mask_expanded] = 0.0

        # 3. Combine Features & Project
        x = torch.cat([x_seq, x_loop, x_dist, x_paired], dim=-1)
        x = self.input_proj(x)
        x = self.dropout(x)

        # 4. Backbone Processing
        for layer in self.layers:
            x = layer(x)

        # 5. Heads
        reg_out = self.reg_head(x)
        recon_out = self.recon_head(x)

        return reg_out, recon_out
