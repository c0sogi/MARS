import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import ModelConfig


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Encodes signed distances using continuous sinusoidal functions.
    Cite solution_lesson_node_00042: Prefer Sinusoidal Encodings Over RBF for 1D Sequence Distances.
    Cite solution_lesson_node_00024: Preserve Directionality in Structural Distance Encodings.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Standard Transformer frequencies: 10000^(-2i/d)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(np.log(10000.0) / dim))
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        """
        Args:
            x: (Batch, Seq_Len) tensor of signed distances.
        Returns:
            (Batch, Seq_Len, dim) tensor of positional embeddings.
        """
        # (B, L, 1) * (1, 1, dim/2) -> (B, L, dim/2)
        args = x.unsqueeze(-1) * self.div_term.view(1, 1, -1)

        emb = torch.zeros(x.size(0), x.size(1), self.dim, device=x.device)
        emb[..., 0::2] = torch.sin(args)
        emb[..., 1::2] = torch.cos(args)
        return emb


class WideResBiGRU(nn.Module):
    """
    A Residual Bidirectional GRU block that maintains a 'Wide Stream' of features.
    Input dimension is 2*H, and the BiGRU output (2*H) is added to the input.
    Uses Pre-LayerNorm configuration.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        # The stream width is maintained at 2 * hidden_dim
        self.stream_dim = 2 * hidden_dim

        # Pre-LayerNorm
        self.norm = nn.LayerNorm(self.stream_dim)

        # Bidirectional GRU
        # Input: 2*H
        # Hidden (per direction): H -> Output (concatenated): 2*H
        self.gru = nn.GRU(
            input_size=self.stream_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        """
        Args:
            x: (Batch, Seq_Len, 2*hidden_dim)
        Returns:
            output: (Batch, Seq_Len, 2*hidden_dim)
        """
        residual = x

        # Pre-Norm
        out = self.norm(x)

        # BiGRU
        out, _ = self.gru(out)

        # Residual Connection
        return residual + out


class RNARegressor(nn.Module):
    """
    Main architecture: Sinusoidal-Encoded Deep Residual BiGRU.
    Cite solution_lesson_node_00041: Residual Stream Width in Bidirectional Recurrent Networks.
    Cite solution_lesson_node_00033: Avoid global pooling/aggregation (removed ScalarMixture).
    """

    def __init__(self, config=ModelConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers

        # --- 1. Embeddings ---
        # Nucleotide Embedding (A, G, U, C)
        self.seq_embed = nn.Embedding(4, self.hidden_dim // 2)

        # Loop Type Embedding (S, M, I, B, H, E, X)
        self.loop_embed = nn.Embedding(7, self.hidden_dim // 2)

        # Sinusoidal Distance Encoding
        # We project the embedding dimension (hidden_dim // 2) directly
        self.dist_embed = SinusoidalPositionalEmbedding(self.hidden_dim // 2)

        # --- 2. Wide Stream Projection ---
        # Concatenated Input: (H/2) [Seq] + (H/2) [Loop] + (H/2) [Dist] = 1.5 * H
        # Project to Wide Stream: 2 * H
        input_feat_dim = (self.hidden_dim // 2) * 3
        self.input_proj = nn.Linear(input_feat_dim, 2 * self.hidden_dim)

        # --- 3. Backbone (Stack of WideResBiGRU) ---
        self.layers = nn.ModuleList(
            [WideResBiGRU(self.hidden_dim) for _ in range(self.num_layers)]
        )

        # --- 4. Output Head ---
        # Projects from wide stream (2*H) to 3 target channels
        self.head = nn.Linear(2 * self.hidden_dim, 3)

    def forward(self, seq, loop, dist, mask):
        """
        Args:
            seq: (B, L) LongTensor - Sequence tokens
            loop: (B, L) LongTensor - Loop type tokens
            dist: (B, L) FloatTensor - Signed pair distances
            mask: (B, L) FloatTensor - 1.0 if paired, 0.0 if unpaired
        Returns:
            logits: (B, L, 3) FloatTensor - Predicted degradation rates
        """
        # Embeddings
        x_seq = self.seq_embed(seq)  # (B, L, H/2)
        x_loop = self.loop_embed(loop)  # (B, L, H/2)

        # Distance Features
        x_dist = self.dist_embed(dist)  # (B, L, H/2)

        # Apply mask to distance features (zero out features for unpaired bases)
        x_dist = x_dist * mask.unsqueeze(-1)

        # Concatenate all features
        x = torch.cat([x_seq, x_loop, x_dist], dim=-1)  # (B, L, 1.5*H)

        # Project to Wide Stream
        x = self.input_proj(x)  # (B, L, 2*H)

        # Pass through Residual Backbone
        curr = x
        for layer in self.layers:
            curr = layer(curr)

        # Final Projection (using only the final layer output)
        logits = self.head(curr)  # (B, L, 3)

        return logits
