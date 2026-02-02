import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResidualBiGRUBlock(nn.Module):
    """
    A Pre-LayerNorm Residual BiGRU Block designed for stability in wide networks.
    Structure: Input -> LayerNorm -> BiGRU -> Dropout -> Residual Add
    """

    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_Len, Hidden_Dim)
        Returns:
            torch.Tensor: Output tensor of shape (Batch, Seq_Len, Hidden_Dim)
        """
        residual = x

        # Pre-LayerNorm
        out = self.ln(x)

        # BiGRU
        # Output shape: (Batch, Seq_Len, Hidden_Dim)
        out, _ = self.gru(out)

        # Dropout
        out = self.dropout(out)

        # Residual Connection
        return residual + out


class ScalarMixture(nn.Module):
    """
    Computes a learnable weighted sum of layer outputs.
    Global Static Aggregation: weights are scalars shared across all samples/positions.
    """

    def __init__(self, n_layers):
        super().__init__()
        self.n_layers = n_layers
        # Initialize weights to zeros (resulting in equal softmax probability initially)
        self.weights = nn.Parameter(torch.zeros(n_layers))

    def forward(self, tensors):
        """
        Args:
            tensors (list[torch.Tensor]): List of L tensors, each (Batch, Seq, Hidden)
        Returns:
            torch.Tensor: Weighted sum tensor (Batch, Seq, Hidden)
        """
        # Stack tensors: (Batch, Seq, Hidden, n_layers)
        stacked = torch.stack(tensors, dim=-1)

        # Compute normalized weights
        norm_weights = F.softmax(self.weights, dim=0)

        # Weighted sum along the last dimension
        # Broadcasting: (Batch, Seq, Hidden, n_layers) * (n_layers)
        weighted_sum = torch.sum(stacked * norm_weights, dim=-1)

        return weighted_sum


class StabilizedWideBiGRU(nn.Module):
    """
    Topologically-Augmented Stabilized Wide-Stream BiGRU.
    Fuses atomic sequence, loop type, pairing distance, and RWPE features.
    Uses a deep, wide BiGRU backbone with Pre-LayerNorm and Gradient Clipping support.
    """

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------------------------
        # 1. Embeddings & Input Processing
        # ----------------------------------------------------------------------
        # Atomic Sequence Embedding
        self.seq_embed = nn.Embedding(Config.VOCAB_SIZE_SEQ, Config.EMBED_DIM_SEQ)

        # Predicted Loop Type Embedding
        self.loop_embed = nn.Embedding(Config.VOCAB_SIZE_LOOP, Config.EMBED_DIM_LOOP)

        # Calculate Total Input Dimension for Early Fusion
        # Seq (128) + Loop (64) + Pair (64) + RWPE (5) = 261
        self.input_dim = (
            Config.EMBED_DIM_SEQ
            + Config.EMBED_DIM_LOOP
            + Config.EMBED_DIM_PAIR
            + len(Config.RWPE_STEPS)
        )

        # ----------------------------------------------------------------------
        # 2. Recurrent Stem
        # ----------------------------------------------------------------------
        # Projects fused features to the wide residual stream (512)
        self.stem = nn.GRU(
            input_size=self.input_dim,
            hidden_size=Config.HIDDEN_DIM // 2,
            bidirectional=True,
            batch_first=True,
        )

        # ----------------------------------------------------------------------
        # 3. Wide-Stream Backbone
        # ----------------------------------------------------------------------
        # Stack of Residual BiGRU Blocks
        self.blocks = nn.ModuleList(
            [
                ResidualBiGRUBlock(Config.HIDDEN_DIM, Config.DROPOUT)
                for _ in range(Config.NUM_LAYERS)
            ]
        )

        # ----------------------------------------------------------------------
        # 4. Aggregation & Output
        # ----------------------------------------------------------------------
        # Mix outputs from Stem + 6 Blocks (Total 7 layers)
        self.mixture = ScalarMixture(Config.NUM_LAYERS + 1)

        # Final projection to targets
        self.head = nn.Linear(Config.HIDDEN_DIM, Config.NUM_TARGETS)

    def forward(self, seq, loop, rwpe, pair_enc):
        """
        Args:
            seq (torch.Tensor): (Batch, Seq_Len) - Sequence IDs
            loop (torch.Tensor): (Batch, Seq_Len) - Loop Type IDs
            rwpe (torch.Tensor): (Batch, Seq_Len, n_steps) - Random Walk probabilities
            pair_enc (torch.Tensor): (Batch, Seq_Len, pair_dim) - Sinusoidal Pair Encodings

        Returns:
            torch.Tensor: (Batch, Seq_Len, 3) - Predicted values
        """
        # 1. Embed Discrete Features
        emb_seq = self.seq_embed(seq)  # (B, L, 128)
        emb_loop = self.loop_embed(loop)  # (B, L, 64)

        # 2. Early Fusion (Concatenation)
        # Concatenate: Seq(128) + Loop(64) + Pair(64) + RWPE(5)
        x = torch.cat([emb_seq, emb_loop, pair_enc, rwpe], dim=-1)

        # 3. Stem Processing
        stem_out, _ = self.stem(x)  # (B, L, 512)

        # 4. Backbone Processing
        # Collect outputs for scalar mixture
        layer_outputs = [stem_out]
        current_state = stem_out

        for block in self.blocks:
            current_state = block(current_state)
            layer_outputs.append(current_state)

        # 5. Aggregation
        # Mix Stem + 6 Blocks
        agg_repr = self.mixture(layer_outputs)  # (B, L, 512)

        # 6. Output Head
        logits = self.head(agg_repr)  # (B, L, 3)

        return logits
