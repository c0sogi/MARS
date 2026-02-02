import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class HeterogeneousEmbeddings(nn.Module):
    """
    Fuses atomic sequence embeddings, predicted loop type embeddings,
    and fixed sinusoidal pairing encodings.
    """

    def __init__(self):
        super().__init__()
        self.seq_embed = nn.Embedding(4, Config.EMBED_DIM_SEQ)
        self.loop_embed = nn.Embedding(7, Config.EMBED_DIM_LOOP)
        # Pair encoding is passed as a float tensor, so no embedding layer needed for it.

    def forward(self, seq, loop, pair_enc):
        """
        Args:
            seq (torch.LongTensor): (B, L) Sequence tokens.
            loop (torch.LongTensor): (B, L) Loop type tokens.
            pair_enc (torch.FloatTensor): (B, L, Embed_Pair) Sinusoidal encodings.

        Returns:
            torch.FloatTensor: (B, L, Total_Embed_Dim) Fused representation.
        """
        # Embed discrete tokens
        seq_emb = self.seq_embed(seq)  # (B, L, 128)
        loop_emb = self.loop_embed(loop)  # (B, L, 64)

        # Concatenate all features
        # Output dim: 128 + 64 + 64 = 256
        x = torch.cat([seq_emb, loop_emb, pair_enc], dim=-1)
        return x


class VectorScaledResidualBlock(nn.Module):
    """
    A Wide-Stream Residual Block with Channel-Wise Scaling.
    Structure: x + Lambda * Dropout(BiGRU(LayerNorm(x)))
    """

    def __init__(self, d_model, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        # BiGRU maintaining the stream width
        # Hidden size is d_model // 2 per direction -> Total output is d_model
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model // 2,
            bidirectional=True,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        # Learnable Vector Scaling (Diagonal Matrix)
        # Initialized to 1.0 (Identity) for stability
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        residual = x

        # Pre-LayerNorm
        out = self.norm(x)

        # BiGRU Processing
        out, _ = self.gru(out)

        # Dropout in residual branch
        out = self.dropout(out)

        # Vector Scaling (element-wise multiplication)
        out = out * self.scale

        return residual + out


class ScalarMixture(nn.Module):
    """
    Aggregates outputs from multiple layers using a learnable weighted sum.
    """

    def __init__(self, n_layers):
        super().__init__()
        # Initialize weights to zeros (Softmax will make them equal initially)
        self.weights = nn.Parameter(torch.zeros(n_layers))

    def forward(self, tensors):
        """
        Args:
            tensors (list of torch.Tensor): List of tensors, each (B, L, D).

        Returns:
            torch.Tensor: (B, L, D) Weighted sum.
        """
        # Stack tensors: (B, L, D, n_layers)
        stacked = torch.stack(tensors, dim=-1)

        # Compute normalized weights
        norm_weights = F.softmax(self.weights, dim=0)

        # Weighted sum along the last dimension
        # Broadcasting: (B, L, D, n_layers) * (n_layers,)
        weighted_sum = torch.sum(stacked * norm_weights, dim=-1)

        return weighted_sum


class VectorScaledWideStreamBiGRU(nn.Module):
    """
    High-Capacity Wide-Stream BiGRU with Vector Scaling.
    """

    def __init__(self):
        super().__init__()

        # 1. Input Processing
        self.embeddings = HeterogeneousEmbeddings()
        input_dim = Config.EMBED_DIM_SEQ + Config.EMBED_DIM_LOOP + Config.EMBED_DIM_PAIR

        # 2. Stem
        # Projects fused input (256) to residual stream width (512)
        # No dropout here to preserve projection fidelity
        self.stem = nn.GRU(
            input_size=input_dim,
            hidden_size=Config.HIDDEN_DIM // 2,
            bidirectional=True,
            batch_first=True,
        )

        # 3. Backbone (6 Blocks)
        # Maintains width 512 throughout
        self.blocks = nn.ModuleList(
            [
                VectorScaledResidualBlock(Config.HIDDEN_DIM, Config.DROPOUT)
                for _ in range(Config.NUM_LAYERS)
            ]
        )

        # 4. Aggregation
        # Aggregates Stem + 6 Blocks = 7 sources
        self.mixture = ScalarMixture(1 + Config.NUM_LAYERS)

        # 5. Output Head
        self.head = nn.Linear(Config.HIDDEN_DIM, Config.NUM_TARGETS)

    def forward(self, seq, loop, pair_enc):
        # Embed Inputs
        x = self.embeddings(seq, loop, pair_enc)

        # Stem Projection
        x, _ = self.stem(x)

        # Collect outputs for aggregation
        layer_outputs = [x]

        # Pass through Backbone
        for block in self.blocks:
            x = block(x)
            layer_outputs.append(x)

        # Aggregate Representations
        x_agg = self.mixture(layer_outputs)

        # Predict Targets
        logits = self.head(x_agg)

        return logits
