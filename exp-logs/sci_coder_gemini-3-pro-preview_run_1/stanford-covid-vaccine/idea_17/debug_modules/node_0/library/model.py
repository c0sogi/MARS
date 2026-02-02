import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import ModelConfig


class GaussianRBFLayer(nn.Module):
    """
    Projects signed distances into a continuous embedding space using a bank of
    Gaussian Radial Basis Functions with learnable centers and widths.
    """

    def __init__(self, num_rbf, max_dist=110):
        super().__init__()
        self.num_rbf = num_rbf
        # Initialize centers uniformly spanning the range [-max_dist, max_dist]
        centers = torch.linspace(-max_dist, max_dist, num_rbf)
        self.centers = nn.Parameter(centers)

        # Initialize widths (sigma) proportional to the spacing between centers
        sigma = torch.ones(num_rbf) * (2 * max_dist / num_rbf)
        self.sigma = nn.Parameter(sigma)

    def forward(self, dists):
        """
        Args:
            dists: (Batch, Seq_Len) tensor of signed distances.
        Returns:
            (Batch, Seq_Len, num_rbf) tensor of RBF features.
        """
        # Expand dimensions for broadcasting: (B, L, 1) vs (1, 1, num_rbf)
        d = dists.unsqueeze(-1)
        c = self.centers.view(1, 1, -1)
        s = self.sigma.view(1, 1, -1)

        # Gaussian RBF: exp( - (x - mu)^2 / sigma^2 )
        rbf = torch.exp(-((d - c) ** 2) / (s**2))
        return rbf


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


class ScalarMixtureHead(nn.Module):
    """
    Aggregates hidden states from multiple layers using learnable scalar weights.
    Implements a 'Gradient Superhighway' from the loss to all layers.
    """

    def __init__(self, num_layers):
        super().__init__()
        # We aggregate (num_layers) residual blocks + 1 initial projection
        self.num_states = num_layers + 1
        self.weights = nn.Parameter(torch.zeros(self.num_states))

    def forward(self, states):
        """
        Args:
            states: List of tensors, each (Batch, Seq_Len, Dim). Length = num_layers + 1.
        Returns:
            Weighted sum tensor (Batch, Seq_Len, Dim).
        """
        # Softmax normalization ensures weights sum to 1
        w = F.softmax(self.weights, dim=0)

        # Weighted sum
        out = 0
        for i, state in enumerate(states):
            out = out + w[i] * state

        return out


class RNARegressor(nn.Module):
    """
    Main architecture: RBF-Encoded Deep Residual BiGRU with Layer Aggregation.
    """

    def __init__(self, config=ModelConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.num_rbf = config.num_rbf

        # --- 1. Embeddings ---
        # Nucleotide Embedding (A, G, U, C)
        self.seq_embed = nn.Embedding(4, self.hidden_dim // 2)

        # Loop Type Embedding (S, M, I, B, H, E, X)
        self.loop_embed = nn.Embedding(7, self.hidden_dim // 2)

        # RBF Distance Encoding
        self.rbf_layer = GaussianRBFLayer(self.num_rbf)
        self.rbf_proj = nn.Linear(self.num_rbf, self.hidden_dim // 2)

        # --- 2. Wide Stream Projection ---
        # Concatenated Input: (H/2) [Seq] + (H/2) [Loop] + (H/2) [RBF] = 1.5 * H
        # Project to Wide Stream: 2 * H
        input_feat_dim = (self.hidden_dim // 2) * 3
        self.input_proj = nn.Linear(input_feat_dim, 2 * self.hidden_dim)

        # --- 3. Backbone (Stack of WideResBiGRU) ---
        self.layers = nn.ModuleList(
            [WideResBiGRU(self.hidden_dim) for _ in range(self.num_layers)]
        )

        # --- 4. Layer Aggregation ---
        self.aggregator = ScalarMixtureHead(self.num_layers)

        # --- 5. Output Head ---
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

        # RBF Features
        x_rbf = self.rbf_layer(dist)  # (B, L, num_rbf)
        x_rbf = self.rbf_proj(x_rbf)  # (B, L, H/2)

        # Apply mask to RBF features (zero out features for unpaired bases)
        x_rbf = x_rbf * mask.unsqueeze(-1)

        # Concatenate all features
        x = torch.cat([x_seq, x_loop, x_rbf], dim=-1)  # (B, L, 1.5*H)

        # Project to Wide Stream
        x = self.input_proj(x)  # (B, L, 2*H)

        # Store states for aggregation (including the initial projection)
        states = [x]

        # Pass through Residual Backbone
        curr = x
        for layer in self.layers:
            curr = layer(curr)
            states.append(curr)

        # Aggregate states
        out_agg = self.aggregator(states)  # (B, L, 2*H)

        # Final Projection
        logits = self.head(out_agg)  # (B, L, 3)

        return logits
