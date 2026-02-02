import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import Config


class SinusoidalSignedPositionalEncoding(nn.Module):
    """
    Fixed Sinusoidal Encodings for signed distances.
    Preserves sign information to distinguish upstream/downstream dependencies
    by applying sinusoidal functions to signed float values.
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        # Calculate the division term for the sinusoidal formulas
        # div_term = 1 / (10000^(2i/d_model))
        # We compute this once and register it as a buffer
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
        # x shape: [Batch, Seq_Len] -> [Batch, Seq_Len, 1] for broadcasting
        x_expanded = x.unsqueeze(-1)

        # Compute phase: pos / 10000^(2i/d)
        # phase shape: [Batch, Seq_Len, d_model/2]
        phase = x_expanded * self.div_term

        # Create output tensor
        pe = torch.zeros(x.size(0), x.size(1), self.d_model, device=x.device)

        # Apply Sin to even indices, Cos to odd indices
        # Sin(-x) = -Sin(x), preserving sign info
        # Cos(-x) = Cos(x), symmetric
        pe[:, :, 0::2] = torch.sin(phase)
        pe[:, :, 1::2] = torch.cos(phase)

        return pe


class RNAModel(nn.Module):
    """
    Noise-Stabilized High-Capacity Wide-Stream BiGRU.
    Features:
    - Proportional Embeddings (Seq, Loop, PairDist)
    - Continuous Noise Injection
    - High-Fidelity Stem (No Dropout)
    - Wide-Stream Residual Backbone (Pre-LN, 512 width)
    - Scalar Mixture Aggregation
    """

    def __init__(self):
        super().__init__()

        # =====================================================================
        # 1. Embeddings
        # =====================================================================
        self.emb_seq = nn.Embedding(4, Config.EMB_DIM_SEQ)  # A, G, C, U
        self.emb_loop = nn.Embedding(7, Config.EMB_DIM_LOOP)  # Loop types
        self.emb_pair = SinusoidalSignedPositionalEncoding(Config.EMB_DIM_PAIR)

        # Total embedding dimension: 128 + 64 + 64 = 256
        self.emb_total_dim = (
            Config.EMB_DIM_SEQ + Config.EMB_DIM_LOOP + Config.EMB_DIM_PAIR
        )

        # =====================================================================
        # 2. Noise Injection
        # =====================================================================
        self.noise_std = Config.NOISE_STD

        # =====================================================================
        # 3. High-Fidelity Recurrent Stem
        # =====================================================================
        # Projects 256 -> 512. No dropout.
        self.stem_gru = nn.GRU(
            input_size=self.emb_total_dim,
            hidden_size=Config.HIDDEN_DIM // 2,  # Bidirectional, so /2
            batch_first=True,
            bidirectional=True,
        )

        # =====================================================================
        # 4. Backbone: Wide-Stream Residual Blocks
        # =====================================================================
        self.num_layers = Config.NUM_LAYERS
        self.blocks = nn.ModuleList()

        for _ in range(self.num_layers):
            # Pre-LayerNorm configuration
            ln = nn.LayerNorm(Config.HIDDEN_DIM)

            # Wide BiGRU (Input 512 -> Output 512)
            gru = nn.GRU(
                input_size=Config.HIDDEN_DIM,
                hidden_size=Config.HIDDEN_DIM // 2,
                batch_first=True,
                bidirectional=True,
            )

            # Inter-layer Dropout
            drop = nn.Dropout(Config.DROPOUT)

            self.blocks.append(nn.ModuleDict({"ln": ln, "gru": gru, "drop": drop}))

        # =====================================================================
        # 5. Aggregation: Scalar Mixture
        # =====================================================================
        # Learnable weights for Stem + 6 Blocks
        self.mix_weights = nn.Parameter(torch.zeros(self.num_layers + 1))

        # =====================================================================
        # 6. Output Head
        # =====================================================================
        # Shared projection to 3 targets
        self.head = nn.Linear(Config.HIDDEN_DIM, 3)

    def forward(self, sequence, loop_type, pair_dist):
        """
        Args:
            sequence: (Batch, Seq_Len) LongTensor
            loop_type: (Batch, Seq_Len) LongTensor
            pair_dist: (Batch, Seq_Len) FloatTensor
        """
        # --- Embeddings ---
        x_seq = self.emb_seq(sequence)  # (B, L, 128)
        x_loop = self.emb_loop(loop_type)  # (B, L, 64)
        x_pair = self.emb_pair(pair_dist)  # (B, L, 64)

        # Concatenate features
        x = torch.cat([x_seq, x_loop, x_pair], dim=-1)  # (B, L, 256)

        # --- Continuous Noise Injection ---
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        # --- Stem ---
        stem_out, _ = self.stem_gru(x)  # (B, L, 512)

        # Initialize list for scalar mixture with stem output
        layer_outputs = [stem_out]
        curr_x = stem_out

        # --- Backbone Blocks ---
        for block in self.blocks:
            residual = curr_x

            # Pre-LayerNorm
            out = block["ln"](curr_x)

            # BiGRU
            out, _ = block["gru"](out)

            # Dropout
            out = block["drop"](out)

            # Residual Addition
            curr_x = residual + out

            layer_outputs.append(curr_x)

        # --- Scalar Mixture Aggregation ---
        # Stack all outputs: (B, L, Hidden, Num_Layers+1)
        stacked_outputs = torch.stack(layer_outputs, dim=-1)

        # Normalize weights via Softmax
        norm_weights = F.softmax(self.mix_weights, dim=0)

        # Weighted Sum
        # Broadcast weights across Batch, Len, Hidden dims
        aggregated = torch.sum(stacked_outputs * norm_weights, dim=-1)  # (B, L, 512)

        # --- Output Head ---
        logits = self.head(aggregated)  # (B, L, 3)

        return logits
