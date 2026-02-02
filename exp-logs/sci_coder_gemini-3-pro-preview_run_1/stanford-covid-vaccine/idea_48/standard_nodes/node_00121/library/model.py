import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import Config


class SinusoidalSignedPositionalEmbedding(nn.Module):
    """
    Learnable embedding for signed distances.
    Maps integer indices (representing shifted signed distances) to dense vectors.
    """

    def __init__(self, dim):
        super().__init__()
        # Input indices are in [0, 255] (shifted by +128)
        self.embedding = nn.Embedding(256, dim)

    def forward(self, x):
        return self.embedding(x)


class ScalarMixture(nn.Module):
    """
    Computes a learnable weighted sum of layer outputs.
    Allows the model to select the best combination of low-level and high-level features.
    """

    def __init__(self, num_layers):
        super().__init__()
        self.num_layers = num_layers
        # Learnable weights for each layer
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, tensors):
        """
        Args:
            tensors: List of (Batch, Seq, Dim) tensors. Length must match num_layers.
        Returns:
            (Batch, Seq, Dim) weighted sum tensor.
        """
        assert (
            len(tensors) == self.num_layers
        ), f"Expected {self.num_layers} tensors, got {len(tensors)}"

        # Compute normalized weights via softmax
        probs = F.softmax(self.weights, dim=0)

        # Weighted sum
        output = 0
        for i, tensor in enumerate(tensors):
            output = output + probs[i] * tensor

        return output


class ResidualBiLSTMBlock(nn.Module):
    """
    Pre-LayerNorm Wide-Stream Residual BiLSTM Block.
    Maintains the full residual stream width (512) to prevent information bottlenecks.
    Structure: x -> LayerNorm -> BiLSTM -> Dropout -> + -> x
    """

    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # BiLSTM: Hidden size is hidden_dim // 2 so that concatenated output is hidden_dim
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-LayerNorm architecture
        residual = x
        x_norm = self.layer_norm(x)

        # BiLSTM
        # self.bilstm returns (output, (h_n, c_n))
        x_out, _ = self.bilstm(x_norm)

        # Dropout
        x_out = self.dropout(x_out)

        # Residual Connection
        return residual + x_out


class TopologicalWideResBiLSTM(nn.Module):
    """
    Topologically-Augmented Wide-Stream Residual BiLSTM.
    Fuses atomic sequence data with geometric (distance) and topological (RWPE) features,
    processed through a deep, high-capacity recurrent backbone.
    """

    def __init__(self, config: Config):
        super().__init__()

        # =========================================================================
        # 1. Heterogeneous Feature Embeddings
        # =========================================================================
        self.seq_embed = nn.Embedding(4, config.seq_embed_dim)  # A, G, C, U
        self.loop_embed = nn.Embedding(7, config.loop_embed_dim)  # S, M, I, B, H, E, X

        # Fixed Sinusoidal Encoding for signed pairing distances
        self.dist_embed = SinusoidalSignedPositionalEmbedding(config.dist_embed_dim)

        # Projection for Random Walk Structural Fingerprint (RWPE)
        # Input is vector of size len(rwpe_steps) (e.g., 5)
        rwpe_input_dim = len(config.rwpe_steps)
        self.rwpe_proj = nn.Linear(rwpe_input_dim, config.rwpe_embed_dim)

        # Calculate total concatenated input dimension
        self.input_dim = (
            config.seq_embed_dim
            + config.loop_embed_dim
            + config.dist_embed_dim
            + config.rwpe_embed_dim
        )

        self.hidden_dim = config.hidden_dim

        # =========================================================================
        # 2. Recurrent Stem
        # =========================================================================
        # Projects concatenated inputs to the residual stream width (512)
        self.stem = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        # Note: No dropout after stem to preserve signal fidelity

        # =========================================================================
        # 3. Backbone (Residual Blocks)
        # =========================================================================
        self.blocks = nn.ModuleList(
            [
                ResidualBiLSTMBlock(self.hidden_dim, dropout=config.dropout)
                for _ in range(config.n_layers)
            ]
        )

        # =========================================================================
        # 4. Aggregation (Scalar Mixture)
        # =========================================================================
        # We aggregate outputs from the Stem + N Blocks
        self.mixture = ScalarMixture(num_layers=1 + config.n_layers)

        # =========================================================================
        # 5. Output Head
        # =========================================================================
        # Shared projection to 3 targets: reactivity, deg_Mg_pH10, deg_Mg_50C
        self.head = nn.Linear(self.hidden_dim, 3)

    def forward(self, sequence, loop_type, rwpe, distance):
        """
        Args:
            sequence: (B, L) LongTensor - Nucleotide indices
            loop_type: (B, L) LongTensor - Loop type indices
            rwpe: (B, L, K) FloatTensor - Random Walk probabilities
            distance: (B, L) FloatTensor - Signed distances to paired base

        Returns:
            logits: (B, L, 3) FloatTensor - Predicted degradation rates
        """
        # --- Feature Embedding ---
        emb_seq = self.seq_embed(sequence)  # (B, L, 128)
        emb_loop = self.loop_embed(loop_type)  # (B, L, 64)
        emb_dist = self.dist_embed(distance)  # (B, L, 64)
        emb_rwpe = self.rwpe_proj(rwpe)  # (B, L, 32)

        # Concatenate all features
        x = torch.cat([emb_seq, emb_loop, emb_dist, emb_rwpe], dim=-1)  # (B, L, 288)

        # --- Stem ---
        x, _ = self.stem(x)  # (B, L, 512)

        # Collect layer outputs for mixture (starting with Stem output)
        layer_outputs = [x]

        # --- Backbone ---
        for block in self.blocks:
            x = block(x)
            layer_outputs.append(x)

        # --- Aggregation ---
        # Weighted sum of Stem and all Block outputs
        x_agg = self.mixture(layer_outputs)  # (B, L, 512)

        # --- Head ---
        logits = self.head(x_agg)  # (B, L, 3)

        return logits
