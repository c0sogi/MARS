import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import Config


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Encodes signed scalar distances into a continuous vector space using sinusoidal functions.
    Preserves the sign information naturally via the odd (sine) components.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Create the frequency constants
        # We use half the dimension for sin and half for cos
        half_dim = dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim).float() / half_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len) containing signed float distances.
        Returns:
            Tensor of shape (Batch, Seq_Len, Dim).
        """
        # (B, L, 1)
        x_expanded = x.unsqueeze(-1)

        # (B, L, half_dim)
        args = x_expanded * self.inv_freq

        # Apply sin and cos
        # sin(-x) = -sin(x), cos(-x) = cos(x)
        pe_sin = torch.sin(args)
        pe_cos = torch.cos(args)

        # Concatenate to get full dimension
        # (B, L, Dim)
        pe = torch.cat([pe_sin, pe_cos], dim=-1)

        return pe


class WideResBiGRUBlock(nn.Module):
    """
    A Residual BiGRU Block that maintains the 'Wide Stream' width.
    Uses Pre-LayerNorm configuration.
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)

        # To maintain width W in the output of a BiGRU, the hidden size per direction must be W/2.
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, W)
        residual = x

        # Pre-LayerNorm
        out = self.ln(x)

        # BiGRU
        out, _ = self.gru(out)

        # Dropout
        out = self.dropout(out)

        # Residual Connection
        return residual + out


class TALA(nn.Module):
    """
    Token-Adaptive Layer Aggregation.
    Dynamically computes position-specific weights to aggregate hidden states from all hierarchy levels.
    """

    def __init__(self, hidden_dim, num_sources):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_sources = num_sources

        # Lightweight MLP to predict mixing weights based on the final layer's context
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, num_sources),
        )

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: List of tensors, each of shape (B, L, W).
                           Represents [stem_out, layer_1, ..., layer_N].
        Returns:
            Aggregated tensor of shape (B, L, W).
        """
        # Stack states along a new dimension: (B, L, Num_Sources, W)
        stack = torch.stack(hidden_states, dim=2)

        # Use the deepest representation (final layer) as the query for context
        final_state = hidden_states[-1]  # (B, L, W)

        # Predict logits for weights: (B, L, Num_Sources)
        scores = self.attention_mlp(final_state)

        # Normalize to get valid attention weights
        weights = F.softmax(scores, dim=-1)

        # Expand for broadcasting: (B, L, Num_Sources, 1)
        weights_expanded = weights.unsqueeze(-1)

        # Weighted sum over the source dimension
        # (B, L, Num_Sources, W) * (B, L, Num_Sources, 1) -> Sum over dim 2
        aggregated = torch.sum(stack * weights_expanded, dim=2)

        return aggregated


class TokenAdaptiveWideResBiGRU(nn.Module):
    """
    Main Architecture: Token-Adaptive Wide-Stream Residual BiGRU.
    Combines multi-modal embeddings, a deep residual recurrent backbone, and
    dynamic hierarchical aggregation.
    """

    def __init__(self):
        super().__init__()

        # 1. Input Embeddings
        self.seq_embed = nn.Embedding(Config.VOCAB_SIZE, Config.EMBED_DIM)
        self.loop_embed = nn.Embedding(Config.LOOP_VOCAB_SIZE, Config.LOOP_EMBED_DIM)
        self.dist_embed = SinusoidalPositionalEmbedding(Config.DIST_EMBED_DIM)

        # Concatenated Input Dimension
        input_dim = Config.EMBED_DIM + Config.LOOP_EMBED_DIM + Config.DIST_EMBED_DIM

        # 2. Recurrent Stem
        # Projects inputs to the Wide Stream width (HIDDEN_DIM)
        self.stem = nn.GRU(
            input_size=input_dim,
            hidden_size=Config.HIDDEN_DIM // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.stem_dropout = nn.Dropout(Config.DROPOUT)

        # 3. Backbone: Wide-Stream Residual Blocks
        self.layers = nn.ModuleList(
            [
                WideResBiGRUBlock(Config.HIDDEN_DIM, Config.DROPOUT)
                for _ in range(Config.NUM_LAYERS)
            ]
        )

        # 4. Token-Adaptive Layer Aggregation
        # Sources = Stem + N Layers
        num_sources = 1 + Config.NUM_LAYERS
        self.tala = TALA(Config.HIDDEN_DIM, num_sources)

        # 5. Output Head
        # Shared projection for all targets
        self.head = nn.Linear(Config.HIDDEN_DIM, Config.NUM_TARGETS)

    def forward(self, sequence, loop_type, pair_dist):
        """
        Args:
            sequence: (B, L) LongTensor
            loop_type: (B, L) LongTensor
            pair_dist: (B, L) FloatTensor (Signed distances)
        Returns:
            (B, L, 3) FloatTensor containing predictions
        """
        # --- Embeddings ---
        emb_seq = self.seq_embed(sequence)  # (B, L, E_seq)
        emb_loop = self.loop_embed(loop_type)  # (B, L, E_loop)
        emb_dist = self.dist_embed(pair_dist)  # (B, L, E_dist)

        # Concatenate features
        x = torch.cat([emb_seq, emb_loop, emb_dist], dim=-1)  # (B, L, Input_Dim)

        # --- Stem ---
        x, _ = self.stem(x)  # (B, L, W)
        x = self.stem_dropout(x)

        # Initialize list of hidden states with the stem output
        hidden_states = [x]

        # --- Backbone ---
        for layer in self.layers:
            x = layer(x)
            hidden_states.append(x)

        # --- Aggregation (TALA) ---
        # Dynamically aggregate information from all depths
        agg_state = self.tala(hidden_states)  # (B, L, W)

        # --- Head ---
        out = self.head(agg_state)  # (B, L, 3)

        return out
