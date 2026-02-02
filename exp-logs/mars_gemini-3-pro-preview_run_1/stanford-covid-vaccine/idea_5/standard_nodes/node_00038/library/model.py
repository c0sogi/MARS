import torch
import torch.nn as nn
import math
from library.config import (
    VOCAB_SIZE_SEQ,
    VOCAB_SIZE_LOOP,
    EMBED_DIM_SEQ,
    EMBED_DIM_LOOP,
    EMBED_DIM_DIST,
    HIDDEN_DIM_GRU,
    NUM_LAYERS_GRU,
    DROPOUT_GRU,
    NUM_TARGETS,
)


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Encodes scalar distances using sinusoidal functions.
    Preserves sign information (directionality).
    """

    def __init__(self, d_model):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.d_model = d_model
        # Create a buffer for the denominator term
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        """
        Args:
            x: (Batch, Seq_Len) - Float tensor of signed distances
        Returns:
            (Batch, Seq_Len, d_model)
        """
        # x.unsqueeze(-1) -> (B, L, 1)
        # div_term -> (d_model/2,)
        # phase -> (B, L, d_model/2)
        phase = x.unsqueeze(-1) * self.div_term

        # Interleave sin and cos
        # Shape: (B, L, d_model)
        pe = torch.zeros(x.size(0), x.size(1), self.d_model, device=x.device)
        pe[:, :, 0::2] = torch.sin(phase)
        pe[:, :, 1::2] = torch.cos(phase)
        return pe


class RNAResidualBiGRU(nn.Module):
    """
    Residual Bidirectional GRU with Pre-LayerNorm.
    """

    def __init__(self):
        super(RNAResidualBiGRU, self).__init__()

        # 1. Embeddings
        self.embed_seq = nn.Embedding(VOCAB_SIZE_SEQ, EMBED_DIM_SEQ)
        self.embed_loop = nn.Embedding(VOCAB_SIZE_LOOP, EMBED_DIM_LOOP)
        self.embed_dist = SinusoidalPositionalEmbedding(EMBED_DIM_DIST)

        # Input projection to match hidden dimension
        input_dim = EMBED_DIM_SEQ + EMBED_DIM_LOOP + EMBED_DIM_DIST
        self.hidden_dim = 2 * HIDDEN_DIM_GRU  # BiGRU output size
        self.proj_in = nn.Linear(input_dim, self.hidden_dim)

        # 2. Residual BiGRU Layers
        self.layers = nn.ModuleList()
        for _ in range(NUM_LAYERS_GRU):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(self.hidden_dim),
                        "gru": nn.GRU(
                            input_size=self.hidden_dim,
                            hidden_size=HIDDEN_DIM_GRU,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True,
                        ),
                        "dropout": nn.Dropout(DROPOUT_GRU),
                    }
                )
            )

        # 3. Output Head
        self.fc_out = nn.Linear(self.hidden_dim, NUM_TARGETS)

    def forward(self, x_seq, x_dist, x_loop):
        """
        Args:
            x_seq: (B, L)
            x_dist: (B, L)
            x_loop: (B, L)
        """
        # Embed
        emb_s = self.embed_seq(x_seq)
        emb_l = self.embed_loop(x_loop)
        emb_d = self.embed_dist(x_dist)

        # Concat
        x = torch.cat([emb_s, emb_l, emb_d], dim=-1)

        # Project to hidden dim
        x = self.proj_in(x)

        # Residual Blocks (Pre-LN)
        for layer in self.layers:
            # x + Block(Norm(x))
            x_norm = layer["norm"](x)
            out, _ = layer["gru"](x_norm)
            x = x + layer["dropout"](out)

        # Output
        out = self.fc_out(x)
        return out


def weighted_masked_mse_loss(preds, targets, masks, weights):
    """
    Calculates the Mean Squared Error, masked to valid positions.
    Ignores sample weights to avoid overfitting to clean data (Cite Lesson 00011).
    """
    # Squared Error: (B, L, 5)
    squared_error = (preds - targets) ** 2

    # Apply Mask (broadcast over last dim): (B, L, 5)
    masked_squared_error = squared_error * masks.unsqueeze(-1)

    # Total error sum
    total_loss = masked_squared_error.sum()

    # Normalization: Count of valid positions
    normalization = masks.sum() * targets.shape[-1]

    # Avoid division by zero
    epsilon = 1e-8
    loss = total_loss / (normalization + epsilon)

    return loss
