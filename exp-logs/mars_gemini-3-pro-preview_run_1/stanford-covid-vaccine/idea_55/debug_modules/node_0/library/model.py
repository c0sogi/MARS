import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import Config


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Encodes signed scalar values (pair distances) into a high-dimensional vector
    using fixed sinusoidal functions. Preserves sign and magnitude information.
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Create constant 'div_term' for the frequencies
        # div_term = 1 / (10000 ^ (2i / d_model))
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len) containing signed float distances.
        Returns:
            Tensor of shape (Batch, Seq_Len, d_model)
        """
        # x is (Batch, Seq_Len) -> (Batch, Seq_Len, 1)
        x_unsqueeze = x.unsqueeze(-1)

        # Calculate sine and cosine components
        # (Batch, Seq_Len, 1) * (d_model/2) -> (Batch, Seq_Len, d_model/2)
        phase = x_unsqueeze * self.div_term

        pe_sin = torch.sin(phase)
        pe_cos = torch.cos(phase)

        # Interleave sin and cos: [sin, cos, sin, cos, ...]
        # Stack along last dim and flatten
        pe = torch.cat([pe_sin, pe_cos], dim=-1)

        return pe


class ScaledResidualBlock(nn.Module):
    """
    A Residual Block containing a Pre-LayerNorm BiGRU with Learnable Residual Scaling.
    Structure: x + alpha * Dropout(BiGRU(LayerNorm(x)))
    """

    def __init__(self, d_model, dropout=0.0, init_scale=1.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        # BiGRU: Hidden size is d_model // 2 so that output is d_model
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model // 2,
            bidirectional=True,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        # Learnable scalar for residual scaling, initialized to init_scale
        self.alpha = nn.Parameter(torch.tensor(init_scale))

    def forward(self, x):
        """
        Args:
            x: Input tensor (Batch, Seq_Len, d_model)
        Returns:
            Output tensor (Batch, Seq_Len, d_model)
        """
        residual = x

        # Pre-LayerNorm
        out = self.norm(x)

        # BiGRU
        # self.gru returns (output, h_n). We only need output.
        out, _ = self.gru(out)

        # Dropout
        out = self.dropout(out)

        # Scaled Residual Connection
        return residual + self.alpha * out


class ScalarMixture(nn.Module):
    """
    Computes a learnable weighted sum of N tensors.
    Weights are normalized via Softmax to ensure stability.
    """

    def __init__(self, n_layers):
        super().__init__()
        self.n_layers = n_layers
        # Initialize weights to 0 (resulting in equal softmax probability initially)
        self.weights = nn.Parameter(torch.zeros(n_layers))

    def forward(self, tensors):
        """
        Args:
            tensors: List of N tensors, each shape (Batch, Seq_Len, d_model)
        Returns:
            Weighted sum tensor (Batch, Seq_Len, d_model)
        """
        if len(tensors) != self.n_layers:
            raise ValueError(f"Expected {self.n_layers} tensors, got {len(tensors)}")

        # Normalize weights
        norm_weights = F.softmax(self.weights, dim=0)

        # Compute weighted sum
        # Stack tensors to (N, Batch, Seq_Len, d_model) for efficient broadcasting
        stacked = torch.stack(tensors, dim=0)

        # Reshape weights for broadcasting: (N, 1, 1, 1)
        w_broadcast = norm_weights.view(-1, 1, 1, 1)

        weighted_sum = torch.sum(w_broadcast * stacked, dim=0)

        return weighted_sum


class ScaledResidualWideStreamBiGRU(nn.Module):
    """
    Main Architecture:
    1. Heterogeneous Embeddings (Seq, Loop, PairDist)
    2. BiGRU Stem (Projection to 512 dim)
    3. 6 Scaled Residual Blocks (Width 512)
    4. Scalar Mixture Aggregation
    5. Output Head
    """

    def __init__(self):
        super().__init__()

        # 1. Embeddings
        self.seq_embed = nn.Embedding(Config.VOCAB_SIZE_SEQ, Config.EMBED_DIM_SEQ)
        self.loop_embed = nn.Embedding(Config.VOCAB_SIZE_LOOP, Config.EMBED_DIM_LOOP)
        self.dist_embed = SinusoidalPositionalEmbedding(Config.EMBED_DIM_PAIR)

        total_input_dim = (
            Config.EMBED_DIM_SEQ + Config.EMBED_DIM_LOOP + Config.EMBED_DIM_PAIR
        )

        # 2. Stem
        # Projects concatenated embeddings to HIDDEN_DIM (512)
        # No dropout after stem (Node 00109)
        self.stem = nn.GRU(
            input_size=total_input_dim,
            hidden_size=Config.HIDDEN_DIM // 2,  # Bidirectional -> 512 total
            bidirectional=True,
            batch_first=True,
        )

        # 3. Backbone
        self.blocks = nn.ModuleList(
            [
                ScaledResidualBlock(
                    d_model=Config.HIDDEN_DIM,
                    dropout=Config.DROPOUT,
                    init_scale=Config.INIT_LAYER_SCALE,
                )
                for _ in range(Config.NUM_LAYERS)
            ]
        )

        # 4. Aggregation
        # We aggregate the output of the Stem + outputs of all Blocks
        # Total layers = 1 (Stem) + NUM_LAYERS
        self.mixture = ScalarMixture(n_layers=1 + Config.NUM_LAYERS)

        # 5. Output Head
        self.head = nn.Linear(Config.HIDDEN_DIM, len(Config.TARGET_COLS))

    def forward(self, sequence, loop_type, pair_dist):
        """
        Args:
            sequence: (Batch, Seq_Len) LongTensor
            loop_type: (Batch, Seq_Len) LongTensor
            pair_dist: (Batch, Seq_Len) FloatTensor
        Returns:
            (Batch, Seq_Len, 3) FloatTensor
        """
        # --- Embeddings ---
        emb_seq = self.seq_embed(sequence)  # (B, L, 128)
        emb_loop = self.loop_embed(loop_type)  # (B, L, 64)
        emb_dist = self.dist_embed(pair_dist)  # (B, L, 64)

        # Concatenate
        x = torch.cat([emb_seq, emb_loop, emb_dist], dim=-1)  # (B, L, 256)

        # --- Stem ---
        x_stem, _ = self.stem(x)  # (B, L, 512)

        # Collect layer outputs for mixture
        layer_outputs = [x_stem]

        # --- Backbone ---
        curr_x = x_stem
        for block in self.blocks:
            curr_x = block(curr_x)
            layer_outputs.append(curr_x)

        # --- Aggregation ---
        x_agg = self.mixture(layer_outputs)  # (B, L, 512)

        # --- Head ---
        logits = self.head(x_agg)  # (B, L, 3)

        return logits
