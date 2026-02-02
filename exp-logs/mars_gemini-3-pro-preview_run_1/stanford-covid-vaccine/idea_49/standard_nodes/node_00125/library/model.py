import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResidualBiGRUBlock(nn.Module):
    """
    A Pre-LayerNorm Residual BiGRU Block.
    Maintains the residual stream width throughout (Wide-Stream design).

    Structure:
    Input -> LayerNorm -> BiGRU -> Dropout -> + -> Output
                                     |
    Input ---------------------------+
    """

    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        # BiGRU: hidden_size is d_model // 2 so the concatenated output is d_model
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model // 2,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Save residual
        residual = x

        # Pre-LayerNorm
        out = self.ln(x)

        # BiGRU
        out, _ = self.gru(out)

        # Dropout
        out = self.dropout(out)

        # Residual Connection
        return residual + out


class ScalarMixture(nn.Module):
    """
    Computes a learnable weighted sum of a list of tensors.
    Weights are normalized via Softmax.
    """

    def __init__(self, num_layers):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, tensors):
        """
        Args:
            tensors (list of torch.Tensor): List of N tensors, each shape (B, L, D).
        Returns:
            torch.Tensor: Weighted sum, shape (B, L, D).
        """
        # Stack tensors: (N, B, L, D)
        stacked = torch.stack(tensors, dim=0)

        # Normalize weights
        norm_weights = F.softmax(self.weights, dim=0)

        # Reshape weights for broadcasting: (N, 1, 1, 1)
        norm_weights = norm_weights.view(-1, 1, 1, 1)

        # Weighted sum
        weighted_sum = torch.sum(stacked * norm_weights, dim=0)

        return weighted_sum


class SpectralTopologicalBiGRU(nn.Module):
    """
    Spectral-Topological Wide-Stream Residual BiGRU.

    Fuses atomic sequence data with spectral graph features (LPE) to capture
    global topology, processed by a deep residual recurrent backbone.
    """

    def __init__(self):
        super().__init__()

        # =====================================================================
        # 1. Feature Embeddings
        # =====================================================================

        # Atomic Sequence Embedding
        self.seq_embed = nn.Embedding(Config.VOCAB_SIZE_SEQ, Config.EMBED_DIM_SEQ)

        # Predicted Loop Type Embedding
        self.loop_embed = nn.Embedding(Config.VOCAB_SIZE_LOOP, Config.EMBED_DIM_LOOP)

        # Laplacian Positional Encoding Projection
        # Projects the k eigenvectors (Config.LPE_DIM) to a dense embedding
        self.lpe_proj = nn.Linear(Config.LPE_DIM, Config.LPE_EMBED_DIM)

        # Note: Pair Distance is already embedded via Sinusoidal Encoding in the dataset
        # with dimension Config.EMBED_DIM_PAIR.

        # Calculate total concatenated input dimension
        # 128 + 64 + 64 + 32 = 288
        input_dim = (
            Config.EMBED_DIM_SEQ
            + Config.EMBED_DIM_LOOP
            + Config.EMBED_DIM_PAIR
            + Config.LPE_EMBED_DIM
        )

        # =====================================================================
        # 2. Stem
        # =====================================================================
        # Projects concatenated features to the residual stream width (384)
        # Strictly BiGRU, no dropout after stem.
        self.stem = nn.GRU(
            input_size=input_dim,
            hidden_size=Config.HIDDEN_DIM // 2,
            bidirectional=True,
            batch_first=True,
        )

        # =====================================================================
        # 3. Backbone
        # =====================================================================
        # Stack of Wide-Stream Residual Blocks
        self.blocks = nn.ModuleList(
            [
                ResidualBiGRUBlock(Config.HIDDEN_DIM, dropout=Config.DROPOUT)
                for _ in range(Config.NUM_LAYERS)
            ]
        )

        # =====================================================================
        # 4. Aggregation
        # =====================================================================
        # Mixes outputs from Stem + 6 Blocks (Total 7 layers)
        self.mixture = ScalarMixture(num_layers=Config.NUM_LAYERS + 1)

        # =====================================================================
        # 5. Output Head
        # =====================================================================
        # Shared projection to targets
        self.head = nn.Linear(Config.HIDDEN_DIM, Config.NUM_TARGETS)

    def forward(self, sequence, loop_type, pair_dist, lpe):
        """
        Args:
            sequence (torch.Tensor): (B, L) LongTensor
            loop_type (torch.Tensor): (B, L) LongTensor
            pair_dist (torch.Tensor): (B, L, 64) FloatTensor (Sinusoidal Encoded)
            lpe (torch.Tensor): (B, L, 8) FloatTensor (Eigenvectors)

        Returns:
            torch.Tensor: (B, L, 3) Predictions
        """
        # 1. Embed Features
        emb_seq = self.seq_embed(sequence)  # (B, L, 128)
        emb_loop = self.loop_embed(loop_type)  # (B, L, 64)
        emb_lpe = self.lpe_proj(lpe)  # (B, L, 32)

        # 2. Concatenate (Early Fusion)
        # pair_dist is already (B, L, 64)
        x = torch.cat([emb_seq, emb_loop, pair_dist, emb_lpe], dim=-1)  # (B, L, 288)

        # 3. Stem
        x, _ = self.stem(x)  # (B, L, 384)

        # Collect outputs for Scalar Mixture
        # We start with the stem output
        layer_outputs = [x]

        # 4. Backbone Blocks
        for block in self.blocks:
            x = block(x)
            layer_outputs.append(x)

        # 5. Aggregation
        # Weighted sum of [Stem, Block1, ..., Block6]
        x_agg = self.mixture(layer_outputs)  # (B, L, 384)

        # 6. Head
        out = self.head(x_agg)  # (B, L, 3)

        return out
