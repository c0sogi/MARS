import torch
import torch.nn as nn
import math
from library.config import Config


class SinusoidalDistanceEmbedding(nn.Module):
    """
    Encodes signed integer distances using fixed sinusoidal functions.
    Preserves directionality (sign) and magnitude.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, distances):
        """
        Args:
            distances (torch.Tensor): (Batch, Seq_Len) float tensor of signed distances.
        Returns:
            torch.Tensor: (Batch, Seq_Len, Embed_Dim)
        """
        device = distances.device
        half_dim = self.embed_dim // 2

        # Compute frequencies: 1 / 10000^(2i/dim)
        # We use log space for numerical stability
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=device) * -emb)

        # Shape broadcasting:
        # distances: (B, L) -> (B, L, 1)
        # emb: (half_dim) -> (1, 1, half_dim)
        # scaled_dist: (B, L, half_dim)
        scaled_dist = distances.unsqueeze(-1) * emb.view(1, 1, -1)

        # Apply sin and cos
        sin_part = torch.sin(scaled_dist)
        cos_part = torch.cos(scaled_dist)

        # Concatenate to form full embedding (B, L, 2*half_dim)
        pos_emb = torch.cat([sin_part, cos_part], dim=-1)

        # Handle odd embed_dim edge case (though Config.EMBED_DIM is 128)
        if self.embed_dim % 2 == 1:
            pos_emb = torch.cat(
                [pos_emb, torch.zeros_like(distances.unsqueeze(-1))], dim=-1
            )

        return pos_emb


class ResidualBiGRU(nn.Module):
    """
    A Pre-LayerNorm Bidirectional GRU block that maintains the residual stream width.
    Structure: x = x + Dropout(BiGRU(LayerNorm(x)))
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        # Bidirectional GRU: hidden_size is halved so output dim matches input dim
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-LayerNorm
        residual = x
        out = self.layer_norm(x)

        # GRU pass
        # out shape: (Batch, Seq, Hidden_Dim)
        out, _ = self.gru(out)

        # Dropout
        out = self.dropout(out)

        # Residual connection
        return residual + out


class ChannelWeightedAggregator(nn.Module):
    """
    Learns a weight vector for each layer to compute a channel-wise weighted sum.
    Allows specific features (channels) to be selected from specific depths.
    """

    def __init__(self, num_layers, hidden_dim):
        super().__init__()
        # Learnable weights: (Num_Layers, Hidden_Dim)
        # Initialized to 0 to start with uniform weighting (after softmax)
        self.weights = nn.Parameter(torch.zeros(num_layers, hidden_dim))

    def forward(self, hidden_states):
        """
        Args:
            hidden_states (list of torch.Tensor): List of L+1 tensors, each (Batch, Seq, Hidden).
        Returns:
            torch.Tensor: Aggregated tensor (Batch, Seq, Hidden).
        """
        # Stack tensors: (Batch, Seq, Num_Layers, Hidden)
        stacked = torch.stack(hidden_states, dim=2)

        # Compute Softmax over the layer dimension (dim 0 of weights)
        # weights: (L, D) -> softmax -> (L, D)
        norm_weights = torch.softmax(self.weights, dim=0)

        # Reshape for broadcasting: (1, 1, L, D)
        norm_weights = norm_weights.view(1, 1, *norm_weights.shape)

        # Weighted Sum
        # (B, S, L, D) * (1, 1, L, D) -> Sum over L -> (B, S, D)
        aggregated = torch.sum(stacked * norm_weights, dim=2)

        return aggregated


class RNAModel(nn.Module):
    """
    Channel-Weighted Wide-Stream Residual BiGRU Model.
    """

    def __init__(self, config=Config):
        super().__init__()

        # 1. Input Embeddings
        self.seq_embed = nn.Embedding(len(config.TOKEN_VOCAB), config.EMBED_DIM)
        self.loop_embed = nn.Embedding(len(config.LOOP_VOCAB), config.EMBED_DIM)
        self.dist_embed = SinusoidalDistanceEmbedding(config.EMBED_DIM)

        # Concatenated input dimension
        input_dim = config.EMBED_DIM * 3

        # 2. Recurrent Stem
        # Projects concatenated inputs to the residual stream width (512)
        self.stem_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=config.HIDDEN_DIM // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.stem_dropout = nn.Dropout(config.DROPOUT)

        # 3. Backbone: Stack of Residual BiGRU blocks
        self.layers = nn.ModuleList(
            [
                ResidualBiGRU(config.HIDDEN_DIM, config.DROPOUT)
                for _ in range(config.NUM_LAYERS)
            ]
        )

        # 4. Aggregator
        # Aggregates Stem output + All Layer outputs
        self.aggregator = ChannelWeightedAggregator(
            num_layers=config.NUM_LAYERS + 1, hidden_dim=config.HIDDEN_DIM
        )

        # 5. Output Head
        # Shared linear projection for all targets
        self.head = nn.Linear(config.HIDDEN_DIM, config.NUM_TARGETS)

    def forward(self, seq, loop, dist):
        """
        Args:
            seq (torch.Tensor): (Batch, Seq_Len) - Sequence tokens
            loop (torch.Tensor): (Batch, Seq_Len) - Loop type tokens
            dist (torch.Tensor): (Batch, Seq_Len) - Signed pair distances
        Returns:
            torch.Tensor: (Batch, Seq_Len, Num_Targets)
        """
        # Generate Embeddings
        s_emb = self.seq_embed(seq)  # (B, L, E)
        l_emb = self.loop_embed(loop)  # (B, L, E)
        d_emb = self.dist_embed(dist)  # (B, L, E)

        # Concatenate
        x = torch.cat([s_emb, l_emb, d_emb], dim=-1)  # (B, L, 3E)

        # Pass through Stem
        x, _ = self.stem_gru(x)
        x = self.stem_dropout(x)

        # Collect states for aggregation (start with Stem output)
        all_hidden_states = [x]

        # Pass through Residual Blocks
        for layer in self.layers:
            x = layer(x)
            all_hidden_states.append(x)

        # Aggregate features from all depths
        x_final = self.aggregator(all_hidden_states)

        # Final Projection
        logits = self.head(x_final)

        return logits
