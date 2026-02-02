import torch
import torch.nn as nn
import math
from library.config import Config


class SignedSinusoidalEmbedding(nn.Module):
    """
    Encodes signed integer distances using sinusoidal functions.
    Preserves directionality via the odd/even properties of sin/cos.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Ensure dim is even for sin/cos pairing
        if dim % 2 != 0:
            raise ValueError(
                f"Embedding dimension {dim} must be even for Sinusoidal Embedding"
            )

        # Pre-compute division term: 10000^(-2i/dim)
        # shape: (dim/2,)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        """
        Args:
            x (torch.LongTensor): (Batch, Seq_Len) Signed integer distances.
        Returns:
            torch.FloatTensor: (Batch, Seq_Len, Dim) Sinusoidal embeddings.
        """
        # x.unsqueeze(-1): (B, L, 1)
        # self.div_term: (D/2,)
        # arg: (B, L, D/2)
        arg = x.unsqueeze(-1).float() * self.div_term

        # Initialize output tensor
        pe = torch.zeros(*x.shape, self.dim, device=x.device)

        # Apply sin to even indices, cos to odd indices
        pe[..., 0::2] = torch.sin(arg)
        pe[..., 1::2] = torch.cos(arg)

        return pe


class ScalarMixture(nn.Module):
    """
    Computes a learnable weighted sum of a list of tensors.
    """

    def __init__(self, num_tensors):
        super().__init__()
        # Initialize weights to zeros (resulting in equal softmax probability initially)
        self.weights = nn.Parameter(torch.zeros(num_tensors))

    def forward(self, tensors):
        """
        Args:
            tensors (list of torch.Tensor): List of N tensors, each (B, L, W).
        Returns:
            torch.Tensor: Weighted sum (B, L, W).
        """
        # Stack tensors: (N, B, L, W)
        stacked = torch.stack(tensors, dim=0)

        # Compute normalized weights: (N,)
        norm_weights = torch.softmax(self.weights, dim=0)

        # Reshape for broadcasting: (N, 1, 1, 1)
        norm_weights = norm_weights.view(-1, 1, 1, 1)

        # Weighted sum
        weighted_sum = (stacked * norm_weights).sum(dim=0)

        return weighted_sum


class ResidualBiGRUBlock(nn.Module):
    """
    Wide-Stream Residual Block with Pre-LayerNorm and BiGRU.
    Maintains the residual stream width W without bottlenecks.
    """

    def __init__(self, width, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(width)

        # BiGRU hidden size is width/2, so output size is width/2 * 2 = width
        self.gru = nn.GRU(
            input_size=width,
            hidden_size=width // 2,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Residual connection path
        residual = x

        # Pre-Norm
        out = self.ln(x)

        # BiGRU
        out, _ = self.gru(out)

        # Dropout
        out = self.dropout(out)

        # Add Residual
        return residual + out


class RNAModel(nn.Module):
    """
    Scalar-Aggregated Wide-Stream Residual BiGRU Network.
    """

    def __init__(self):
        super().__init__()

        # =====================================================================
        # Hyperparameters
        # =====================================================================
        embed_dim = Config.EMBED_DIM
        hidden_dim = Config.HIDDEN_DIM  # H
        width = 2 * hidden_dim  # W (Wide stream width)
        num_layers = Config.NUM_LAYERS
        dropout = Config.DROPOUT

        # =====================================================================
        # Input Embeddings
        # =====================================================================
        self.seq_embed = nn.Embedding(Config.VOCAB_SIZE_SEQ, embed_dim)
        self.loop_embed = nn.Embedding(Config.VOCAB_SIZE_LOOP, embed_dim)
        self.dist_embed = SignedSinusoidalEmbedding(embed_dim)

        # Concatenated dimension: 3 * embed_dim
        input_dim = 3 * embed_dim

        # =====================================================================
        # Recurrent Stem (Structural Correction)
        # =====================================================================
        # Projects inputs to the residual stream width W
        self.stem = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.stem_dropout = nn.Dropout(dropout)

        # =====================================================================
        # Backbone: Wide-Stream Residual Blocks
        # =====================================================================
        self.blocks = nn.ModuleList(
            [ResidualBiGRUBlock(width, dropout) for _ in range(num_layers)]
        )

        # =====================================================================
        # Aggregation & Output
        # =====================================================================
        # We aggregate outputs from Stem + All Blocks (1 + num_layers)
        self.mixture = ScalarMixture(num_layers + 1)

        # Final projection to targets
        self.head = nn.Linear(width, Config.NUM_TARGETS)

    def forward(self, seq, loop, dist):
        """
        Args:
            seq (torch.LongTensor): (B, L) Sequence indices.
            loop (torch.LongTensor): (B, L) Loop type indices.
            dist (torch.LongTensor): (B, L) Signed pair distances.

        Returns:
            torch.Tensor: (B, L, 3) Predicted values for scored columns.
        """
        # 1. Embed Inputs
        s = self.seq_embed(seq)  # (B, L, E)
        l = self.loop_embed(loop)  # (B, L, E)
        d = self.dist_embed(dist)  # (B, L, E)

        # Concatenate features
        x = torch.cat([s, l, d], dim=-1)  # (B, L, 3*E)

        # 2. Recurrent Stem
        x, _ = self.stem(x)
        x = self.stem_dropout(x)  # (B, L, W)

        # Collect outputs for aggregation
        layer_outputs = [x]

        # 3. Backbone Blocks
        for block in self.blocks:
            x = block(x)
            layer_outputs.append(x)

        # 4. Scalar Mixture Aggregation
        # Weighted sum of all layers
        x_agg = self.mixture(layer_outputs)  # (B, L, W)

        # 5. Output Head
        logits = self.head(x_agg)  # (B, L, 3)

        return logits
