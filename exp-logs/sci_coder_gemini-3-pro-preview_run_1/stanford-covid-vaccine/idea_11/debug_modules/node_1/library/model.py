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


class StructureAugmentedHybridNetwork(nn.Module):
    """
    Structure-Augmented Hybrid Recurrent-Attention Network.

    Architecture:
    1. Embeddings: Sequence + Loop + Pair Distance + Paired Base Identity (Teleportation).
    2. Stage 1: Deep Pre-LayerNorm Residual BiGRU backbone (Sequential Bias).
    3. Stage 2: Transformer Encoder (Global Refinement).
    4. Head: Projection to target columns.
    """

    def __init__(self, config=Config):
        super().__init__()
        self.hidden_dim = config.HIDDEN_DIM

        # 1. Embeddings
        self.seq_emb = nn.Embedding(config.VOCAB_SIZE, self.hidden_dim)
        self.loop_emb = nn.Embedding(config.LOOP_VOCAB_SIZE, self.hidden_dim)

        # 2. Distance Encoding (Sinusoidal)
        # Max distance is bounded by sequence length. Adding a buffer for safety.
        self.dist_emb = SinusoidalPositionalEncoding(
            self.hidden_dim, max_len=config.SEQ_LENGTH + 20
        )

        # 3. Paired-Base Identity (Teleportation)
        # Learnable embedding for unpaired bases (where no paired neighbor exists)
        self.unpaired_emb = nn.Parameter(torch.randn(1, 1, self.hidden_dim))

        # 4. Input Projection
        # Concatenate: Seq(H) + Loop(H) + Dist(H) + PairedSeq(H) = 4H
        self.input_proj = nn.Sequential(
            nn.Linear(4 * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(config.DROPOUT),
        )

        # 5. Stage 1: Pre-LN BiGRU Stack
        # We use a ModuleList to manually implement the Residual + Pre-LN connection
        self.gru_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(config.NUM_GRU_LAYERS):
            self.norms.append(nn.LayerNorm(self.hidden_dim))
            self.gru_layers.append(
                nn.GRU(
                    self.hidden_dim,
                    self.hidden_dim // 2,  # Bidirectional output will be hidden_dim
                    batch_first=True,
                    bidirectional=True,
                )
            )

        # 6. Stage 2: Transformer Encoder Refinement
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=config.NHEAD,
            dim_feedforward=self.hidden_dim * 4,
            dropout=config.DROPOUT,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.NUM_TRANSFORMER_LAYERS
        )

        # 7. Output Head
        # Predicts the 3 scored targets: reactivity, deg_Mg_pH10, deg_Mg_50C
        self.head = nn.Linear(self.hidden_dim, 3)

    def forward(self, seq, loop, pair_index):
        """
        Args:
            seq (torch.Tensor): (B, L) Integer encoded sequence.
            loop (torch.Tensor): (B, L) Integer encoded loop type.
            pair_index (torch.Tensor): (B, L) Indices of paired bases, -1 if unpaired.
        """
        B, L = seq.shape
        device = seq.device

        # --- Feature Generation ---

        # 1. Base Embeddings
        x_seq = self.seq_emb(seq)  # (B, L, H)
        x_loop = self.loop_emb(loop)  # (B, L, H)

        # 2. Distance Embeddings
        # Create grid of indices [0, 1, ..., L-1]
        indices = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

        # Identify paired positions
        is_paired = pair_index != -1

        # Calculate distance: |i - j| where paired, else 0
        dist = torch.zeros_like(indices)
        dist[is_paired] = torch.abs(indices[is_paired] - pair_index[is_paired])

        x_dist = self.dist_emb(dist)  # (B, L, H)

        # 3. Paired-Base Identity (Teleportation)
        # We want to gather the sequence embedding of the paired base j for position i.

        # Prepare gather indices: replace -1 with 0 to avoid index out of bounds errors
        gather_indices = pair_index.clone()
        gather_indices[~is_paired] = 0

        # Expand indices for gathering across the hidden dimension
        gather_indices_expanded = gather_indices.unsqueeze(-1).expand(
            -1, -1, self.hidden_dim
        )

        # Gather operation: x_paired[b, i, :] = x_seq[b, pair_index[b, i], :]
        x_paired_raw = torch.gather(x_seq, 1, gather_indices_expanded)

        # Apply mask: Use gathered embedding if paired, else use special unpaired token
        x_paired = torch.where(
            is_paired.unsqueeze(-1), x_paired_raw, self.unpaired_emb.expand(B, L, -1)
        )

        # --- Model Backbone ---

        # Combine all features
        x = torch.cat([x_seq, x_loop, x_dist, x_paired], dim=-1)  # (B, L, 4H)
        x = self.input_proj(x)  # (B, L, H)

        # Stage 1: Residual Pre-LN BiGRU
        for norm, gru in zip(self.norms, self.gru_layers):
            x_norm = norm(x)
            x_gru, _ = gru(x_norm)
            x = x + x_gru  # Residual connection

        # Stage 2: Transformer Refinement
        x = self.transformer(x)

        # Output Projection
        logits = self.head(x)  # (B, L, 3)

        return logits
