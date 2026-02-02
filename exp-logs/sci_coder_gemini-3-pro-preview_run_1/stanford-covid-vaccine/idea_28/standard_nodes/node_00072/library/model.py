import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import (
    NUM_TOKENS,
    NUM_LOOP_TYPES,
    EMBED_DIM,
    HIDDEN_DIM,
    NUM_LAYERS,
    DROPOUT,
    NOISE_SIGMA,
    NUM_TARGETS,
)


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Encodes scalar distances into a high-dimensional vector using sinusoidal functions.
    Handles signed distances by preserving the sign in the input before transformation,
    though standard PE is symmetric, the unique scalar input allows the network to distinguish
    magnitude. To strictly distinguish sign, the raw distance is passed, and the network
    learns from the phase shifts inherent in the manifold or we rely on the distinct
    values.

    Standard PE formula:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        scale = math.log(10000.0) / (half_dim - 1) if half_dim > 1 else 1.0
        # Compute the division term: exp(arange * -scale)
        div_term = torch.exp(torch.arange(half_dim, dtype=torch.float) * -scale)
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        """
        Args:
            x: (Batch, Seq_Len) tensor of float distances.
        Returns:
            (Batch, Seq_Len, Dim) tensor.
        """
        # x shape: (B, L) -> (B, L, 1)
        x_expanded = x.unsqueeze(-1)

        # div_term shape: (half_dim,)
        # args shape: (B, L, half_dim)
        args = x_expanded * self.div_term

        # Compute sin and cos
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        # If dim is odd, pad with zeros (though usually dim is even)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


class ResidualBiGRUBlock(nn.Module):
    """
    A residual block containing LayerNorm, BiGRU, and Dropout.
    Standard Deep Residual RNN architecture.
    Cite solution_lesson_node_00047
    """

    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)

        # BiGRU: Hidden size is half of total width per direction
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        # Pre-Norm
        out = self.norm(x)

        # BiGRU
        out, _ = self.gru(out)

        # Dropout
        out = self.dropout(out)

        # Residual Connection
        return residual + out


class RNAModel(nn.Module):
    """
    Scalar-Aggregated Wide-Stream BiGRU.
    """

    def __init__(self):
        super().__init__()

        # 1. Embeddings
        self.seq_emb = nn.Embedding(NUM_TOKENS, EMBED_DIM)
        self.loop_emb = nn.Embedding(NUM_LOOP_TYPES, EMBED_DIM)
        self.dist_emb = SinusoidalPositionalEmbedding(EMBED_DIM)

        # Input dimension is concatenation of 3 embeddings
        input_dim = 3 * EMBED_DIM

        # 2. Recurrent Stem
        # Projects input features to the main model width (HIDDEN_DIM)
        self.stem = nn.GRU(
            input_size=input_dim,
            hidden_size=HIDDEN_DIM // 2,
            batch_first=True,
            bidirectional=True,
        )

        # 3. Backbone (Residual Blocks)
        self.blocks = nn.ModuleList(
            [ResidualBiGRUBlock(HIDDEN_DIM, DROPOUT) for _ in range(NUM_LAYERS)]
        )

        # 4. Scalar Mixture Aggregation
        # Learnable weights for [Stem, Block1, ..., BlockN]
        self.mix_weights = nn.Parameter(torch.zeros(NUM_LAYERS + 1))

        # 5. Output Head
        self.head = nn.Linear(HIDDEN_DIM, NUM_TARGETS)

        # Hyperparameters
        self.noise_sigma = NOISE_SIGMA

    def forward(self, sequence, loop_type, pair_dist):
        """
        Args:
            sequence: (B, L) LongTensor
            loop_type: (B, L) LongTensor
            pair_dist: (B, L) FloatTensor
        """
        # --- Embeddings ---
        x_seq = self.seq_emb(sequence)  # (B, L, EMBED_DIM)
        x_loop = self.loop_emb(loop_type)  # (B, L, EMBED_DIM)
        x_dist = self.dist_emb(pair_dist)  # (B, L, EMBED_DIM)

        # Concatenate
        x = torch.cat([x_seq, x_loop, x_dist], dim=-1)  # (B, L, 3*EMBED_DIM)

        # --- Continuous Noise Injection ---
        # Regularization: Add Gaussian noise to continuous embeddings during training
        if self.training and self.noise_sigma > 0:
            noise = torch.randn_like(x) * self.noise_sigma
            x = x + noise

        # --- Stem ---
        x, _ = self.stem(x)  # (B, L, HIDDEN_DIM)

        # Collect outputs for scalar mixture
        layer_outputs = [x]

        # --- Backbone ---
        for block in self.blocks:
            x = block(x)
            layer_outputs.append(x)

        # --- Scalar Mixture Aggregation ---
        # Stack: (B, L, HIDDEN_DIM, Num_Layers+1)
        stacked = torch.stack(layer_outputs, dim=-1)

        # Normalize weights
        weights = F.softmax(self.mix_weights, dim=0)  # (Num_Layers+1,)

        # Weighted Sum: sum(stacked * weights)
        # Broadcasting weights over B, L, HIDDEN_DIM
        x_agg = torch.sum(stacked * weights, dim=-1)  # (B, L, HIDDEN_DIM)

        # --- Head ---
        logits = self.head(x_agg)  # (B, L, NUM_TARGETS)

        return logits
