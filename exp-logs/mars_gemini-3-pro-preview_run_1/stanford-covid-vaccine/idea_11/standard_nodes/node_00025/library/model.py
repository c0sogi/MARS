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
    Distance-Aware Residual BiGRU Network.
    Cite solution_lesson_node_00017: Pre-LayerNorm Facilitates Scaling in Deep Residual RNNs.
    Cite solution_lesson_node_00024: Preserve Directionality in Structural Distance Encodings.
    Cite solution_lesson_node_00023: Implicit Structural Bias via Continuous Distance Encoding Outperforms Explicit Graph Features.
    """

    def __init__(self, config=Config):
        super().__init__()
        self.hidden_dim = config.HIDDEN_DIM
        self.seq_len = config.SEQ_LENGTH

        # 1. Embeddings
        self.seq_emb = nn.Embedding(config.VOCAB_SIZE, self.hidden_dim)
        self.loop_emb = nn.Embedding(config.LOOP_VOCAB_SIZE, self.hidden_dim)

        # 2. Signed Distance Encoding (Sinusoidal)
        # Range of signed distance is roughly [-L, +L].
        # We shift by +L to make indices positive [0, 2L].
        # Max index needed is 2*L + small buffer.
        self.dist_emb = SinusoidalPositionalEncoding(
            self.hidden_dim, max_len=2 * config.SEQ_LENGTH + 20
        )

        # 3. Input Projection
        # Concatenate: Seq(H) + Loop(H) + Dist(H) = 3H
        self.input_proj = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(config.DROPOUT),
        )

        # 4. Deep Pre-LN BiGRU Stack
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

        # 5. Output Head
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

        # 2. Signed Distance Embeddings
        # Create grid of indices [0, 1, ..., L-1]
        indices = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

        # Handle unpaired bases:
        # If paired, target = pair_index.
        # If unpaired (pair_index == -1), target = current_index (distance 0).
        # This creates a "0 distance" for unpaired bases, distinct from +/- distances for paired.
        target_indices = pair_index.clone()
        unpaired_mask = pair_index == -1
        target_indices[unpaired_mask] = indices[unpaired_mask]

        # Calculate signed distance: target - current
        signed_dist = target_indices - indices  # Range approx [-L, L]

        # Shift to positive indices for embedding lookup
        # Add L to center 0 at L.
        dist_indices = signed_dist + self.seq_len

        x_dist = self.dist_emb(dist_indices)  # (B, L, H)

        # --- Model Backbone ---

        # Combine features (Implicit structure via distance)
        x = torch.cat([x_seq, x_loop, x_dist], dim=-1)  # (B, L, 3H)
        x = self.input_proj(x)  # (B, L, H)

        # Residual Pre-LN BiGRU
        for norm, gru in zip(self.norms, self.gru_layers):
            x_norm = norm(x)
            x_gru, _ = gru(x_norm)
            x = x + x_gru  # Residual connection

        # Output Projection
        logits = self.head(x)  # (B, L, 3)

        return logits
