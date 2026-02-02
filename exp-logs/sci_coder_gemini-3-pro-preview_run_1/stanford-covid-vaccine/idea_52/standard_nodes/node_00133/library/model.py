import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ScalarMixture(nn.Module):
    """
    Computes a learnable weighted sum of a list of tensors.
    Used to aggregate outputs from the stem and all backbone blocks.
    """

    def __init__(self, num_layers):
        super().__init__()
        self.num_layers = num_layers
        # Initialize weights to be equal (1/num_layers) initially via softmax
        self.weights = nn.Parameter(torch.zeros(num_layers, dtype=torch.float32))

    def forward(self, tensors):
        """
        Args:
            tensors: List of tensors, each of shape (B, L, D).
                     Length of list must match num_layers.
        Returns:
            Weighted sum tensor of shape (B, L, D).
        """
        assert (
            len(tensors) == self.num_layers
        ), f"Expected {self.num_layers} tensors, got {len(tensors)}"

        # Compute normalized weights
        norm_weights = F.softmax(self.weights, dim=0)

        # Weighted sum
        # Stack tensors to (B, L, D, num_layers) or compute iteratively
        # Iterative is memory efficient
        out = 0
        for i, t in enumerate(tensors):
            out = out + norm_weights[i] * t

        return out


class BiGRUBlock(nn.Module):
    """
    Wide-Stream Residual BiGRU Block.
    Structure: Pre-LayerNorm -> BiGRU -> Dropout -> Residual Add
    """

    def __init__(self, hidden_size, dropout_p=0.2):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)

        # BiGRU: input_size=hidden_size, output_size=hidden_size
        # To get output_size=hidden_size with bidirectional=True,
        # the internal hidden_size must be hidden_size // 2.
        assert hidden_size % 2 == 0, "Hidden size must be divisible by 2 for BiGRU"
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        # x: (B, L, hidden_size)
        residual = x

        # Pre-LayerNorm
        out = self.layer_norm(x)

        # BiGRU
        out, _ = self.gru(out)

        # Dropout
        out = self.dropout(out)

        # Residual Connection
        out = out + residual

        return out


class RNARegressor(nn.Module):
    """
    Position-Aware Proportional Wide-Stream Residual BiGRU.

    Inputs:
        - Sequence Indices (Atomic)
        - Loop Type Indices
        - Pair Distance Embeddings (Sinusoidal, Fixed)
        - Absolute Position Embeddings (Sinusoidal, Fixed)

    Architecture:
        1. Embeddings & Concatenation
        2. Stem BiGRU (Projection to residual width)
        3. 6x Residual BiGRU Blocks
        4. Scalar Mixture Aggregation
        5. Linear Head
    """

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------------------------
        # 1. Input Embeddings
        # ----------------------------------------------------------------------
        self.seq_embedding = nn.Embedding(
            num_embeddings=Config.VOCAB_SIZE_SEQ, embedding_dim=Config.EMB_SEQ_DIM
        )

        self.loop_embedding = nn.Embedding(
            num_embeddings=Config.VOCAB_SIZE_LOOP, embedding_dim=Config.EMB_LOOP_DIM
        )

        # Pair and Pos embeddings are pre-computed fixed sinusoidal features passed as floats.
        # Dimensions are defined in Config: EMB_PAIR_DIM, EMB_POS_DIM

        # Total concatenated input dimension
        # 128 (Seq) + 64 (Loop) + 64 (Pair) + 32 (Pos) = 288
        self.input_dim = Config.INPUT_DIM

        # ----------------------------------------------------------------------
        # 2. Stem
        # ----------------------------------------------------------------------
        # Projects concatenated input to model width (384) via BiGRU
        # No dropout after stem
        assert Config.HIDDEN_SIZE % 2 == 0
        self.stem_gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=Config.HIDDEN_SIZE // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # ----------------------------------------------------------------------
        # 3. Backbone
        # ----------------------------------------------------------------------
        self.blocks = nn.ModuleList(
            [
                BiGRUBlock(hidden_size=Config.HIDDEN_SIZE, dropout_p=Config.DROPOUT)
                for _ in range(Config.NUM_LAYERS)
            ]
        )

        # ----------------------------------------------------------------------
        # 4. Aggregation
        # ----------------------------------------------------------------------
        # Mixes output of Stem + 6 Blocks (Total 7 layers)
        self.mixture = ScalarMixture(num_layers=Config.NUM_LAYERS + 1)

        # ----------------------------------------------------------------------
        # 5. Output Head
        # ----------------------------------------------------------------------
        self.head = nn.Linear(Config.HIDDEN_SIZE, Config.NUM_CLASSES)

    def forward(self, seq_ids, loop_ids, pair_emb, pos_emb):
        """
        Args:
            seq_ids: (B, L) LongTensor
            loop_ids: (B, L) LongTensor
            pair_emb: (B, L, EMB_PAIR_DIM) FloatTensor
            pos_emb: (B, L, EMB_POS_DIM) FloatTensor

        Returns:
            logits: (B, L, NUM_CLASSES)
        """
        # 1. Embeddings
        seq_emb = self.seq_embedding(seq_ids)  # (B, L, 128)
        loop_emb = self.loop_embedding(loop_ids)  # (B, L, 64)

        # 2. Concatenation
        # [Seq, Loop, Pair, Pos] -> (B, L, 288)
        x = torch.cat([seq_emb, loop_emb, pair_emb, pos_emb], dim=-1)

        # 3. Stem
        # (B, L, 288) -> (B, L, 384)
        stem_out, _ = self.stem_gru(x)

        # Collect outputs for mixture
        layer_outputs = [stem_out]

        # 4. Backbone
        current_out = stem_out
        for block in self.blocks:
            current_out = block(current_out)
            layer_outputs.append(current_out)

        # 5. Aggregation
        # Weighted sum of [Stem, Block1, ..., Block6]
        aggregated_out = self.mixture(layer_outputs)

        # 6. Head
        logits = self.head(aggregated_out)

        return logits
