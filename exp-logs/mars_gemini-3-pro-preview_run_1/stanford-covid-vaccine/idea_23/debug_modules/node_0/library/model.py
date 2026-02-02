import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Encodes signed pairing distances using sinusoidal functions.
    Preserves the sign information (upstream vs downstream) via the odd symmetry of the sine function.
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Create constant 'pe' matrix with values dependent on pos and i
        # div_term: exp(arange(0, d_model, 2) * -(log(10000.0) / d_model))
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        """
        Args:
            x: (Batch, Seq_Len) tensor of signed distances (float or int).
        Returns:
            (Batch, Seq_Len, d_model) tensor of positional encodings.
        """
        # x shape: (B, L)
        # Unsqueeze to (B, L, 1) for broadcasting
        phase = x.unsqueeze(-1) * self.div_term.view(1, 1, -1)

        # Calculate sine and cosine
        # sin(-x) = -sin(x), cos(-x) = cos(x)
        # This naturally encodes the sign of the distance
        pe_sin = torch.sin(phase)
        pe_cos = torch.cos(phase)

        # Concatenate along the last dimension
        # Shape: (B, L, d_model) assuming d_model is even
        return torch.cat([pe_sin, pe_cos], dim=-1)


class FeatureFusion(nn.Module):
    """
    Pointwise MLP to mix atomic sequence, loop, and distance embeddings
    before temporal processing.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class ResidualBiGRUBlock(nn.Module):
    """
    Wide-Stream Residual Block.
    Structure: x = x + BiGRU(LayerNorm(x))
    Maintains the full residual stream width.
    """

    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,  # Bidirectional, so output is hidden_size
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.norm(x)
        out, _ = self.gru(out)
        out = self.dropout(out)
        return residual + out


class ScalarMixture(nn.Module):
    """
    Computes a learnable weighted sum of a list of tensors.
    Used to aggregate outputs from different layers of the model.
    """

    def __init__(self, num_layers):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, tensors):
        """
        Args:
            tensors: List of tensors, each shape (Batch, Seq, Hidden)
        Returns:
            Weighted sum tensor of shape (Batch, Seq, Hidden)
        """
        # Stack tensors: (Num_Layers, Batch, Seq, Hidden)
        stacked = torch.stack(tensors, dim=0)

        # Compute softmax weights
        probs = F.softmax(self.weights, dim=0)

        # Broadcast weights: (Num_Layers, 1, 1, 1)
        probs = probs.view(-1, 1, 1, 1)

        # Weighted sum
        return (stacked * probs).sum(dim=0)


class RNAModel(nn.Module):
    """
    SWA-Optimized Pre-Mixed Wide-Stream BiGRU with Global Initialization.
    """

    def __init__(self):
        super().__init__()

        # 1. Embeddings
        self.seq_embed = nn.Embedding(Config.VOCAB_SIZE_SEQ, Config.EMBED_DIM_SEQ)
        self.loop_embed = nn.Embedding(Config.VOCAB_SIZE_LOOP, Config.EMBED_DIM_LOOP)
        self.dist_embed = SinusoidalPositionalEmbedding(Config.EMBED_DIM_DIST)

        # 2. Feature Fusion
        total_embed_dim = (
            Config.EMBED_DIM_SEQ + Config.EMBED_DIM_LOOP + Config.EMBED_DIM_DIST
        )
        self.fusion = FeatureFusion(total_embed_dim, Config.HIDDEN_SIZE)

        # 3. Global Context Initialization
        # Projects global average pool to initial hidden state (h_0)
        # BiGRU h_0 shape: (2, Batch, Hidden//2) -> Total elements: Batch * Hidden
        self.init_proj = nn.Linear(Config.HIDDEN_SIZE, Config.HIDDEN_SIZE)

        # 4. Recurrent Stem
        self.stem = nn.GRU(
            input_size=Config.HIDDEN_SIZE,
            hidden_size=Config.HIDDEN_SIZE // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.stem_dropout = nn.Dropout(Config.DROPOUT)

        # 5. Backbone (Residual Blocks)
        self.layers = nn.ModuleList(
            [
                ResidualBiGRUBlock(Config.HIDDEN_SIZE, Config.DROPOUT)
                for _ in range(Config.NUM_LAYERS)
            ]
        )

        # 6. Output Aggregation
        # Inputs to mixture: Stem output + N Block outputs
        self.mixture = ScalarMixture(num_layers=1 + Config.NUM_LAYERS)

        # 7. Prediction Head
        self.head = nn.Linear(Config.HIDDEN_SIZE, Config.NUM_TARGETS)

    def forward(self, seq, loop, dist):
        """
        Args:
            seq: (Batch, Seq_Len) - Sequence indices
            loop: (Batch, Seq_Len) - Loop type indices
            dist: (Batch, Seq_Len) - Signed pairing distances
        Returns:
            logits: (Batch, Seq_Len, 3)
        """
        # Embed Inputs
        e_seq = self.seq_embed(seq)  # (B, L, E_seq)
        e_loop = self.loop_embed(loop)  # (B, L, E_loop)
        e_dist = self.dist_embed(dist)  # (B, L, E_dist)

        # Concatenate
        cat_features = torch.cat([e_seq, e_loop, e_dist], dim=-1)

        # Fuse Features (Non-linear mixing)
        x = self.fusion(cat_features)  # (B, L, Hidden)

        # Global Initialization
        # Compute Global Average Pool of fused features
        global_feat = x.mean(dim=1)  # (B, Hidden)

        # Project to h_0
        h0 = self.init_proj(global_feat)  # (B, Hidden)

        # Reshape for GRU: (Num_Dirs * Num_Layers, Batch, Hidden_per_Dir)
        # Here Num_Dirs=2, Num_Layers=1 (for the stem)
        batch_size = x.size(0)
        h0 = (
            h0.view(batch_size, 2, Config.HIDDEN_SIZE // 2)
            .permute(1, 0, 2)
            .contiguous()
        )

        # Stem
        x, _ = self.stem(x, h0)
        x = self.stem_dropout(x)

        # Collect outputs for scalar mixture
        layer_outputs = [x]

        # Backbone
        for layer in self.layers:
            x = layer(x)
            layer_outputs.append(x)

        # Aggregate
        x_final = self.mixture(layer_outputs)

        # Predict
        logits = self.head(x_final)

        return logits
