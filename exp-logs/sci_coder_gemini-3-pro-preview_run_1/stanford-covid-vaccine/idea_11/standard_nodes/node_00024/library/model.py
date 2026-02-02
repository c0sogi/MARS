import torch
import torch.nn as nn
import numpy as np
from library.config import Config


class SinusoidalPositionalEncoding(nn.Module):
    """
    Encodes scalar distances using sinusoidal functions.
    Used for the geometric encoding of base-pair distances.
    """

    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Calculate the division term: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Tensor containing distances (integer indices).
        Returns:
            torch.Tensor: Positional embeddings corresponding to the distances.
        """
        # Clamp indices to ensure they fall within the pre-computed range
        indices = x.clamp(0, self.pe.size(0) - 1)
        return self.pe[indices]


class RNAResidualBiGRU(nn.Module):
    """
    Pure Residual BiGRU with Continuous Distance Encoding.
    Optimized based on Lesson 00023 (Simplicity) and Lesson 00017 (Pre-LN Scaling).

    Architecture:
    1. Embeddings: Sequence + Loop + Pair Distance (Sinusoidal).
    2. Backbone: Deep Pre-LayerNorm Residual BiGRU.
    3. Head: Projection to target columns.
    """

    def __init__(self, config=Config):
        super().__init__()
        self.hidden_dim = config.HIDDEN_DIM

        # 1. Embeddings
        self.seq_emb = nn.Embedding(config.VOCAB_SIZE, self.hidden_dim)
        self.loop_emb = nn.Embedding(config.LOOP_VOCAB_SIZE, self.hidden_dim)

        # 2. Distance Encoding (Sinusoidal)
        self.dist_emb = SinusoidalPositionalEncoding(
            self.hidden_dim, max_len=config.SEQ_LENGTH + 20
        )

        # 3. Input Projection
        # Concatenate: Seq(H) + Loop(H) + Dist(H) = 3H
        self.input_proj = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(config.DROPOUT),
        )

        # 4. Deep Pre-LN Residual BiGRU Stack
        self.gru_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(config.NUM_GRU_LAYERS):
            self.norms.append(nn.LayerNorm(self.hidden_dim))
            self.gru_layers.append(
                nn.GRU(
                    self.hidden_dim,
                    self.hidden_dim // 2,  # Bidirectional output sums to hidden_dim
                    batch_first=True,
                    bidirectional=True,
                )
            )

        # 5. Output Head
        self.head = nn.Linear(self.hidden_dim, 3)

    def forward(self, seq, loop, pair_index):
        B, L = seq.shape
        device = seq.device

        # --- Feature Generation ---
        x_seq = self.seq_emb(seq)
        x_loop = self.loop_emb(loop)

        # Distance Encoding
        indices = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        is_paired = pair_index != -1
        dist = torch.zeros_like(indices)
        dist[is_paired] = torch.abs(indices[is_paired] - pair_index[is_paired])
        x_dist = self.dist_emb(dist)

        # Combine
        x = torch.cat([x_seq, x_loop, x_dist], dim=-1)
        x = self.input_proj(x)

        # --- Backbone (Pre-LayerNorm Residual) ---
        for norm, gru in zip(self.norms, self.gru_layers):
            x_norm = norm(x)
            x_gru, _ = gru(x_norm)
            x = x + x_gru

        # --- Head ---
        logits = self.head(x)
        return logits
