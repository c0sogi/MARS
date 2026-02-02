import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import (
    VOCAB_SIZE_SEQ,
    VOCAB_SIZE_LOOP,
    EMBED_DIM,
    HIDDEN_DIM,
    NUM_LAYERS,
    DROPOUT,
    SEQ_SCORED,
)


class SinusoidalDistanceEmbedding(nn.Module):
    """
    Encodes signed scalar distances using fixed sinusoidal positional encodings.
    Preserves the sign of the distance (upstream vs downstream) via the odd symmetry of the sine function.
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Compute the division term: 10000^(2i/d_model)
        # We use log space for numerical stability: exp(2i * -log(10000) / d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        # x: [Batch, Seq_Len] (signed float/int distances)
        # Output: [Batch, Seq_Len, d_model]

        # Unsqueeze to broadcast: [B, L, 1] * [D/2] -> [B, L, D/2]
        arg = x.unsqueeze(-1) * self.div_term

        # Initialize output tensor
        pe = torch.zeros(*x.shape, self.d_model, device=x.device)

        # Apply Sin to even indices, Cos to odd indices
        pe[..., 0::2] = torch.sin(arg)
        pe[..., 1::2] = torch.cos(arg)

        return pe


class ResidualBiGRUBlock(nn.Module):
    """
    Wide-Stream Residual Block with Pre-LayerNorm configuration.
    Structure: Input -> LN -> BiGRU -> Dropout -> Residual Add
    """

    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        # BiGRU: hidden_size is half of hidden_dim per direction to maintain width
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.norm(x)
        out, _ = self.gru(out)
        out = self.dropout(out)
        return residual + out


class ScalarMixture(nn.Module):
    """
    Computes a learnable weighted sum of a list of tensors.
    Used to aggregate representations from different layers.
    """

    def __init__(self, n_layers):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_layers))

    def forward(self, tensors):
        # tensors: List of [Batch, Seq_Len, Hidden_Dim]
        # Stack inputs: [Batch, Seq_Len, Hidden_Dim, N_Layers]
        stacked = torch.stack(tensors, dim=-1)

        # Softmax ensures weights sum to 1
        norm_weights = F.softmax(self.weights, dim=0)

        # Weighted sum along the last dimension
        return torch.sum(stacked * norm_weights, dim=-1)


class RNANet(nn.Module):
    """
    Log-Space Uncertainty-Aware Wide-Stream Residual BiGRU.
    """

    def __init__(self):
        super().__init__()

        # --- 1. Input Embeddings ---
        self.seq_embed = nn.Embedding(VOCAB_SIZE_SEQ, EMBED_DIM)
        self.loop_embed = nn.Embedding(VOCAB_SIZE_LOOP, EMBED_DIM)
        self.dist_embed = SinusoidalDistanceEmbedding(EMBED_DIM)

        # Concatenated input dimension: 128 * 3 = 384
        input_dim = EMBED_DIM * 3

        # --- 2. Recurrent Stem ---
        # Projects inputs to the residual stream width
        self.stem = nn.GRU(
            input_size=input_dim,
            hidden_size=HIDDEN_DIM // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # --- 3. Backbone ---
        # Stack of Residual BiGRU Blocks
        self.blocks = nn.ModuleList(
            [ResidualBiGRUBlock(HIDDEN_DIM, DROPOUT) for _ in range(NUM_LAYERS)]
        )

        # --- 4. Aggregation ---
        # Mixes output of Stem + 6 Blocks (Total 7)
        self.mixture = ScalarMixture(NUM_LAYERS + 1)

        # --- 5. Dual Output Heads ---
        # Value Head: Predicts reactivity, deg_Mg_pH10, deg_Mg_50C
        self.value_head = nn.Linear(HIDDEN_DIM, 3)
        # Uncertainty Head: Predicts log(error) for the same targets
        self.uncertainty_head = nn.Linear(HIDDEN_DIM, 3)

    def forward(self, seq, loop, pair_dist):
        # Embeddings
        x_seq = self.seq_embed(seq)  # [B, L, 128]
        x_loop = self.loop_embed(loop)  # [B, L, 128]
        x_dist = self.dist_embed(pair_dist)  # [B, L, 128]

        # Concatenate
        x = torch.cat([x_seq, x_loop, x_dist], dim=-1)  # [B, L, 384]

        # Stem Processing
        x, _ = self.stem(x)  # [B, L, 384]

        # Collect layer outputs for aggregation
        layer_outputs = [x]

        # Backbone Processing
        for block in self.blocks:
            x = block(x)
            layer_outputs.append(x)

        # Aggregate representations
        x_agg = self.mixture(layer_outputs)  # [B, L, 384]

        # Prediction Heads
        values = self.value_head(x_agg)  # [B, L, 3]
        uncertainties = self.uncertainty_head(x_agg)  # [B, L, 3]

        return values, uncertainties


class HomoscedasticLoss(nn.Module):
    """
    Multi-Task Loss with learnable uncertainty weighting.
    L = 0.5 * exp(-s1) * L_val + 0.5 * exp(-s2) * L_unc + 0.5 * (s1 + s2)
    """

    def __init__(self):
        super().__init__()
        self.s1 = nn.Parameter(torch.zeros(1))  # Weight for Value Loss
        self.s2 = nn.Parameter(torch.zeros(1))  # Weight for Uncertainty Loss

    def forward(self, pred_val, true_val, pred_unc, true_unc):
        # Inputs: [Batch, SEQ_SCORED, 3]

        # Calculate MSE for both tasks
        loss_val = F.mse_loss(pred_val, true_val)
        loss_unc = F.mse_loss(pred_unc, true_unc)

        # Compute weighted loss
        factor1 = torch.exp(-self.s1)
        factor2 = torch.exp(-self.s2)

        loss = (
            (0.5 * factor1 * loss_val)
            + (0.5 * factor2 * loss_unc)
            + (0.5 * (self.s1 + self.s2))
        )

        return loss, loss_val, loss_unc
