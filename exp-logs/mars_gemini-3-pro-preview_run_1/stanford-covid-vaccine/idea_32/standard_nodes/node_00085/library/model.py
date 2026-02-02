import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalDistanceEmbedding(nn.Module):
    """
    Encodes signed distances using fixed sinusoidal positional encodings.
    Preserves the sign of the distance to distinguish upstream/downstream dependencies.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Create the division term for the sinusoidal functions
        # div_term = 10000^(2i/dim)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, dists):
        """
        Args:
            dists: Signed distance tensor of shape (Batch, Seq_Len)
        Returns:
            Tensor of shape (Batch, Seq_Len, dim)
        """
        # dists: (B, L) -> (B, L, 1)
        # div_term: (dim/2)
        # argument: (B, L, dim/2)
        x = dists.unsqueeze(-1) * self.div_term

        # Initialize embedding tensor
        pe = torch.zeros(*dists.shape, self.dim, device=dists.device, dtype=dists.dtype)

        # Apply Sin to even indices, Cos to odd indices
        pe[..., 0::2] = torch.sin(x)
        pe[..., 1::2] = torch.cos(x)

        return pe


class ResidualBlock(nn.Module):
    """
    Wide-Stream Residual Block with Pre-LayerNorm.
    Maintains the full hidden dimension throughout the block.
    """

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # BiGRU: Hidden size is dim // 2 per direction, so output is dim.
        self.bigru = nn.GRU(dim, dim // 2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        # Pre-LayerNorm
        out = self.norm(x)

        # Wide BiGRU
        out, _ = self.bigru(out)

        # Regularization
        out = self.dropout(out)

        # Residual Connection (Cite solution_lesson_node_00084)
        return residual + out


class ScalarMixture(nn.Module):
    """
    Computes a learnable weighted sum of a list of tensors.
    Used to aggregate features from different layers (Stem + Blocks).
    """

    def __init__(self, n_layers):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_layers))

    def forward(self, tensors):
        """
        Args:
            tensors: List of tensors, all with the same shape (B, L, D)
        Returns:
            Weighted sum tensor (B, L, D)
        """
        # Normalize weights using softmax to ensure stability
        norm_weights = F.softmax(self.weights, dim=0)

        weighted_sum = torch.zeros_like(tensors[0])
        for i, t in enumerate(tensors):
            weighted_sum += norm_weights[i] * t

        return weighted_sum


class RNAModel(nn.Module):
    """
    Main model class implementing the LayerScale-Stabilized Wide-Stream Residual BiGRU.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # --- Embeddings ---
        # Atomic Sequence: A, G, C, U (4 tokens)
        self.seq_embed = nn.Embedding(4, config.embed_dim)

        # Predicted Loop Type: S, M, I, B, H, E, X (7 tokens)
        self.loop_embed = nn.Embedding(7, config.embed_dim)

        # Signed Sinusoidal Pairing Distance
        self.dist_embed = SinusoidalDistanceEmbedding(config.embed_dim)

        # Calculate input dimension for the Stem
        # Concatenation of 3 embeddings of size `embed_dim`
        input_dim = config.embed_dim * 3

        # --- Recurrent Stem ---
        # Projects input features to the residual stream width (hidden_dim)
        self.stem = nn.GRU(
            input_dim, config.hidden_dim // 2, batch_first=True, bidirectional=True
        )

        # --- Backbone ---
        # Stack of Residual Blocks
        self.layers = nn.ModuleList(
            [
                ResidualBlock(
                    dim=config.hidden_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )

        # --- Aggregation ---
        # Mixes outputs from Stem + N Blocks
        self.mixture = ScalarMixture(config.n_layers + 1)

        # --- Output Head ---
        # Shared projection to 3 targets: reactivity, deg_Mg_pH10, deg_Mg_50C
        self.head = nn.Linear(config.hidden_dim, 3)

    def forward(self, seq, loop, dist, mask=None):
        """
        Args:
            seq: (Batch, Seq_Len) Integer sequence tokens
            loop: (Batch, Seq_Len) Integer loop type tokens
            dist: (Batch, Seq_Len) Float signed distances
            mask: (Batch, Seq_Len) Optional mask (unused in computation, passed for API compatibility)

        Returns:
            logits: (Batch, Seq_Len, 3) Predicted values
        """
        # 1. Embed Inputs
        s = self.seq_embed(seq)  # (B, L, E)
        l = self.loop_embed(loop)  # (B, L, E)
        d = self.dist_embed(dist)  # (B, L, E)

        # Concatenate features
        x = torch.cat([s, l, d], dim=-1)  # (B, L, 3E)

        # 2. Process via Stem
        x, _ = self.stem(x)  # (B, L, H)

        # Collect outputs for scalar mixture (start with Stem output)
        all_outputs = [x]

        # 3. Process via Backbone
        for layer in self.layers:
            x = layer(x)
            all_outputs.append(x)

        # 4. Aggregate Layers
        x_agg = self.mixture(all_outputs)  # (B, L, H)

        # 5. Project to Targets
        logits = self.head(x_agg)  # (B, L, 3)

        return logits
