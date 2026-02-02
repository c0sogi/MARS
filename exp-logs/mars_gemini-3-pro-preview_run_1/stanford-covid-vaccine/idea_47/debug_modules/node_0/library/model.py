import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import Config


class SinusoidalEncoding(nn.Module):
    """
    Fixed Sinusoidal Encoding that handles signed integers for pairing distances.
    Preserves the sign of the distance by applying sin/cos to the raw signed value.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Pre-compute the frequency constants
        # div_term = 1 / (10000 ^ (2i / dim))
        # We compute terms for half the dimension since we use both sin and cos to form the full embedding
        half_dim = dim // 2
        div_term = torch.exp(
            torch.arange(0, half_dim).float() * (-math.log(10000.0) / half_dim)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        """
        Args:
            x: (Batch, Seq_Len) tensor of signed integer distances.
        Returns:
            (Batch, Seq_Len, Dim) tensor of encodings.
        """
        # x is (B, L)
        # unsqueeze to (B, L, 1) for broadcasting
        x_expanded = x.unsqueeze(-1).float()

        # div_term is (Dim/2,)
        # arguments: x * div_term -> (B, L, Dim/2)
        args = x_expanded * self.div_term

        # Apply sin and cos
        # Note: sin(-x) = -sin(x), cos(-x) = cos(x). The combination preserves uniqueness for signed inputs.
        sin_enc = torch.sin(args)
        cos_enc = torch.cos(args)

        # Concatenate along the last dim -> (B, L, Dim)
        pe = torch.cat([sin_enc, cos_enc], dim=-1)
        return pe


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM).
    Conditions the 'input_dim' features on 'cond_dim' features via affine transformation.
    Used here to modulate Distance embeddings based on Loop Type context.
    """

    def __init__(self, input_dim, cond_dim):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        # Project condition to 2 * input_dim (gamma and beta)
        self.proj = nn.Linear(cond_dim, 2 * input_dim)

        # Initialize weights to 0 and biases such that gamma=1, beta=0
        # This ensures the layer starts as an identity mapping, aiding stability
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            # Set bias for gamma (first half) to 1.0
            self.proj.bias[:input_dim].fill_(1.0)

    def forward(self, x, condition):
        """
        Args:
            x: Input features to modulate (Batch, Seq, Input_Dim)
            condition: Conditioning features (Batch, Seq, Cond_Dim)
        Returns:
            Modulated features (Batch, Seq, Input_Dim)
        """
        # Get gamma and beta
        params = self.proj(condition)  # (B, L, 2 * Input_Dim)
        gamma, beta = torch.split(params, self.input_dim, dim=-1)

        # Apply modulation: gamma * x + beta
        return gamma * x + beta


class ResidualBiGRU(nn.Module):
    """
    Wide-Stream Residual Block with Pre-LayerNorm configuration.
    Structure: Input -> LN -> BiGRU -> Dropout -> Residual Add
    """

    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        # BiGRU: hidden_size = hidden_dim / 2 so output is hidden_dim
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        # Pre-Norm
        out = self.layer_norm(x)

        # BiGRU
        out, _ = self.gru(out)

        # Dropout
        out = self.dropout(out)

        # Residual Connection
        return residual + out


class ScalarMixture(nn.Module):
    """
    Computes a learnable weighted sum of layer outputs (Global Static Aggregation).
    Uses softmax on weights to ensure stability.
    """

    def __init__(self, n_layers):
        super().__init__()
        self.n_layers = n_layers
        # Learnable weights, initialized to 0 (equivalent to uniform distribution after softmax)
        self.weights = nn.Parameter(torch.zeros(n_layers))

    def forward(self, layers):
        """
        Args:
            layers: List of tensors, each (Batch, Seq, Dim)
        Returns:
            Weighted sum tensor (Batch, Seq, Dim)
        """
        # Stack layers: (Batch, Seq, Dim, N_Layers)
        stacked = torch.stack(layers, dim=-1)

        # Compute softmax weights: (N_Layers,)
        norm_weights = F.softmax(self.weights, dim=0)

        # Weighted sum
        # (B, S, D, N) * (N,) -> sum over N
        weighted_sum = torch.sum(stacked * norm_weights, dim=-1)
        return weighted_sum


class RNAModel(nn.Module):
    """
    Context-Modulated Wide-Stream Residual BiGRU Model.
    Integrates heterogeneous embeddings, geometric-context modulation,
    deep residual recurrence, and scalar aggregation.
    """

    def __init__(self, config=Config):
        super().__init__()
        self.config = config

        # =========================================================================
        # 1. Embeddings
        # =========================================================================
        # Atomic Sequence: A, G, C, U (4 tokens)
        self.seq_embed = nn.Embedding(4, config.EMBED_DIM_SEQ)

        # Predicted Loop Type (7 tokens)
        self.loop_embed = nn.Embedding(7, config.EMBED_DIM_LOOP)

        # Signed Pairing Distance (Fixed Sinusoidal)
        self.dist_embed = SinusoidalEncoding(config.EMBED_DIM_DIST)

        # =========================================================================
        # 2. Geometric-Context Modulation
        # =========================================================================
        # Modulate Distance Embedding (Geometric) using Loop Embedding (Context) via FiLM
        self.film = FiLMLayer(
            input_dim=config.EMBED_DIM_DIST, cond_dim=config.EMBED_DIM_LOOP
        )

        # =========================================================================
        # 3. Fusion & Stem
        # =========================================================================
        # Concatenation dimension
        fusion_dim = (
            config.EMBED_DIM_SEQ + config.EMBED_DIM_LOOP + config.EMBED_DIM_DIST
        )

        # Stem BiGRU (Projection to backbone width)
        # No dropout applied after this layer to preserve projection fidelity
        self.stem_gru = nn.GRU(
            input_size=fusion_dim,
            hidden_size=config.HIDDEN_DIM // 2,
            batch_first=True,
            bidirectional=True,
        )

        # =========================================================================
        # 4. Backbone (Wide-Stream Residual Blocks)
        # =========================================================================
        self.layers = nn.ModuleList(
            [
                ResidualBiGRU(config.HIDDEN_DIM, dropout=config.DROPOUT)
                for _ in range(config.N_LAYERS)
            ]
        )

        # =========================================================================
        # 5. Aggregation & Head
        # =========================================================================
        # Mixture of Stem + N_Layers
        self.mixture = ScalarMixture(n_layers=1 + config.N_LAYERS)

        # Shared Linear Projection to Targets
        self.head = nn.Linear(config.HIDDEN_DIM, config.N_OUTPUTS)

    def forward(self, sequence, loop, distance):
        """
        Args:
            sequence: (Batch, Seq_Len) LongTensor
            loop: (Batch, Seq_Len) LongTensor
            distance: (Batch, Seq_Len) LongTensor (Signed)
        Returns:
            logits: (Batch, Seq_Len, N_Outputs) FloatTensor
        """
        # 1. Generate Embeddings
        emb_seq = self.seq_embed(sequence)  # (B, L, 128)
        emb_loop = self.loop_embed(loop)  # (B, L, 64)
        emb_dist = self.dist_embed(distance)  # (B, L, 64)

        # 2. Apply Modulation (FiLM)
        # Condition geometric distance features on structural loop context
        emb_dist_mod = self.film(emb_dist, emb_loop)

        # 3. Feature Fusion
        # Concatenate: Sequence + Loop + Modulated Distance
        x = torch.cat([emb_seq, emb_loop, emb_dist_mod], dim=-1)  # (B, L, 256)

        # 4. Stem Projection
        x, _ = self.stem_gru(x)  # (B, L, 384)

        # Collect outputs for scalar mixture (starting with Stem)
        layer_outputs = [x]

        # 5. Backbone Processing
        for layer in self.layers:
            x = layer(x)
            layer_outputs.append(x)

        # 6. Aggregation
        x_agg = self.mixture(layer_outputs)  # (B, L, 384)

        # 7. Output Head
        logits = self.head(x_agg)  # (B, L, 3)

        return logits
