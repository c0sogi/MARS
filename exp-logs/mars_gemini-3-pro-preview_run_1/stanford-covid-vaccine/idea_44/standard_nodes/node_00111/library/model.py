import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DualStems(nn.Module):
    """
    Dual-Stream Recurrent Stem.
    Processes sequence identity and structural features in parallel before fusion.
    """

    def __init__(self):
        super().__init__()

        # 1. Embeddings
        # Sequence: 4 tokens (A, G, C, U) -> 128 dim
        self.seq_embedding = nn.Embedding(
            num_embeddings=4, embedding_dim=Config.EMB_DIM_SEQ
        )

        # Loop Type: 7 tokens (S, M, I, B, H, E, X) -> 64 dim
        self.loop_embedding = nn.Embedding(
            num_embeddings=7, embedding_dim=Config.EMB_DIM_LOOP
        )

        # 2. Parallel BiGRUs
        # Sequence Stem: Input 128 -> Hidden 96*2 = 192
        self.seq_gru = nn.GRU(
            input_size=Config.EMB_DIM_SEQ,
            hidden_size=Config.STEM_HIDDEN_SIZE,
            batch_first=True,
            bidirectional=True,
        )

        # Structure Stem: Input (Loop 64 + Dist 64) = 128 -> Hidden 96*2 = 192
        struct_input_dim = Config.EMB_DIM_LOOP + Config.EMB_DIM_DIST
        self.struct_gru = nn.GRU(
            input_size=struct_input_dim,
            hidden_size=Config.STEM_HIDDEN_SIZE,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, seq, loop, dist):
        # seq: (B, L)
        # loop: (B, L)
        # dist: (B, L, 64)

        # --- Sequence Stream ---
        x_seq = self.seq_embedding(seq)  # (B, L, 128)
        out_seq, _ = self.seq_gru(x_seq)  # (B, L, 192)

        # --- Structure Stream ---
        x_loop = self.loop_embedding(loop)  # (B, L, 64)
        # Concatenate loop embedding and distance embedding
        x_struct_input = torch.cat([x_loop, dist], dim=-1)  # (B, L, 128)
        out_struct, _ = self.struct_gru(x_struct_input)  # (B, L, 192)

        # --- Fusion ---
        # Concatenate the outputs of both stems
        # Result dim: 192 + 192 = 384
        out_fused = torch.cat([out_seq, out_struct], dim=-1)

        return out_fused


class ResidualBiGRUBlock(nn.Module):
    """
    Wide-Stream Residual Block with Pre-LayerNorm and Dropout.
    """

    def __init__(self):
        super().__init__()
        self.hidden_size = Config.BACKBONE_HIDDEN_SIZE  # 384

        # Pre-LayerNorm
        self.ln = nn.LayerNorm(self.hidden_size)

        # BiGRU
        # Hidden size per direction = 384 // 2 = 192
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size // 2,
            batch_first=True,
            bidirectional=True,
        )

        # Inter-layer Dropout
        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, x):
        # x: (B, L, 384)

        residual = x

        # Pre-Norm
        out = self.ln(x)

        # Transformation
        out, _ = self.gru(out)

        # Dropout
        out = self.dropout(out)

        # Residual Connection
        return residual + out


class ScalarMixture(nn.Module):
    """
    Computes a learnable weighted sum of a list of tensors.
    """

    def __init__(self, num_tensors):
        super().__init__()
        # Initialize weights to zeros (equivalent to uniform attention after softmax)
        self.weights = nn.Parameter(torch.zeros(num_tensors))

    def forward(self, tensors):
        # tensors: List of [B, L, D]
        # Stack to (num_tensors, B, L, D)
        stacked = torch.stack(tensors, dim=0)

        # Compute normalized weights
        norm_weights = F.softmax(self.weights, dim=0)

        # Reshape weights for broadcasting: (num_tensors, 1, 1, 1)
        norm_weights = norm_weights.view(-1, 1, 1, 1)

        # Weighted sum
        weighted_sum = torch.sum(stacked * norm_weights, dim=0)

        return weighted_sum


class DualStreamBiGRU(nn.Module):
    """
    Main Architecture: Dual-Stream Recurrent Fusion Wide-Stream BiGRU.
    """

    def __init__(self):
        super().__init__()

        # 1. Input Stems
        self.stems = DualStems()

        # 2. Backbone
        # Stack of Residual Blocks
        self.blocks = nn.ModuleList(
            [ResidualBiGRUBlock() for _ in range(Config.BACKBONE_LAYERS)]
        )

        # 3. Aggregation
        # Mixes: 1 Stem Output + N Block Outputs
        self.mixture = ScalarMixture(num_tensors=1 + Config.BACKBONE_LAYERS)

        # 4. Output Head
        # Projects aggregated vector (384) to targets (3)
        self.head = nn.Linear(Config.BACKBONE_HIDDEN_SIZE, Config.NUM_TARGETS)

    def forward(self, inputs):
        """
        Args:
            inputs (dict): Dictionary containing:
                - 'seq': (B, L) LongTensor
                - 'loop': (B, L) LongTensor
                - 'dist': (B, L, 64) FloatTensor

        Returns:
            torch.Tensor: Predictions of shape (B, L, 3)
        """
        seq = inputs["seq"]
        loop = inputs["loop"]
        dist = inputs["dist"]

        # --- Dual Stream Processing ---
        # x_stem: (B, L, 384)
        x_stem = self.stems(seq, loop, dist)

        # Collect layer outputs for mixture
        layer_outputs = [x_stem]

        # --- Backbone Processing ---
        x = x_stem
        for block in self.blocks:
            x = block(x)
            layer_outputs.append(x)

        # --- Aggregation ---
        # Weighted sum of all layers
        x_agg = self.mixture(layer_outputs)  # (B, L, 384)

        # --- Output Head ---
        logits = self.head(x_agg)  # (B, L, 3)

        return logits
