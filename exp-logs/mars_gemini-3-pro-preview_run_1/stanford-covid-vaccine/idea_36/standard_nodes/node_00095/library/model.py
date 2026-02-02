import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config, SinusoidalPositionalEmbedding


class ResidualBiGRUBlock(nn.Module):
    """
    A residual block containing a Pre-LayerNorm BiGRU and Dropout.
    Maintains the wide stream width (hidden_dim) throughout.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.gru = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # x: [Batch, SeqLen, HiddenDim]
        res = x
        out = self.norm(x)
        out, _ = self.gru(out)
        out = self.dropout(out)
        return res + out


class ScalarMixture(nn.Module):
    """
    Computes a learnable weighted sum of a list of tensors.
    """

    def __init__(self, n_layers: int):
        super().__init__()
        # Initialize weights to 1.0
        self.weights = nn.Parameter(torch.ones(n_layers))

    def forward(self, tensors):
        """
        Args:
            tensors: List of tensors, each [Batch, SeqLen, HiddenDim]
        Returns:
            Weighted sum tensor [Batch, SeqLen, HiddenDim]
        """
        # Cite solution_lesson_node_00063: Iterative Accumulation vs Vectorized Stacking
        # Use iterative accumulation to save memory
        norm_weights = F.softmax(self.weights, dim=0)
        out = torch.zeros_like(tensors[0])
        for i, t in enumerate(tensors):
            out += norm_weights[i] * t
        return out


class WideResBiGRU(nn.Module):
    """
    Wide-Stream Residual BiGRU Architecture.

    Reverted to single-task regression to avoid destructive interference from
    unscaled auxiliary targets (Cite solution_lesson_node_00094).

    Features:
    - Multi-channel input embeddings (Sequence, Loop, Distance).
    - Wide-stream residual backbone (BiGRU).
    - Scalar mixture aggregation of all layers.
    """

    def __init__(self, config: Config = None):
        super().__init__()
        if config is None:
            config = Config()
        self.config = config

        # 1. Embeddings
        self.seq_embed = nn.Embedding(4, config.embed_dim)
        self.loop_embed = nn.Embedding(7, config.embed_dim)
        self.dist_embed = SinusoidalPositionalEmbedding(config.embed_dim)

        # Input Projection: Concatenated Embeddings -> Hidden Dim
        self.input_proj = nn.Linear(config.embed_dim * 3, config.hidden_dim)

        # 2. Recurrent Stem
        self.stem_gru = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )

        # 3. Backbone: Residual Blocks
        self.blocks = nn.ModuleList(
            [ResidualBiGRUBlock(config) for _ in range(config.n_layers)]
        )

        # 4. Aggregation: Scalar Mixture
        self.mixture = ScalarMixture(n_layers=config.n_layers + 1)

        # 5. Output Head
        self.value_head = nn.Linear(config.hidden_dim, 3)

    def forward(self, seq, loop, dist):
        # Embeddings
        e_seq = self.seq_embed(seq)
        e_loop = self.loop_embed(loop)
        e_dist = self.dist_embed(dist)

        # Concatenate and Project
        x = torch.cat([e_seq, e_loop, e_dist], dim=-1)
        x = self.input_proj(x)

        # Stem Processing
        x, _ = self.stem_gru(x)

        # Collect layer outputs for mixture (starting with Stem output)
        layer_outputs = [x]

        # Backbone Processing
        curr = x
        for block in self.blocks:
            curr = block(curr)
            layer_outputs.append(curr)

        # Aggregation
        aggregated = self.mixture(layer_outputs)

        # Head
        values = self.value_head(aggregated)

        return values
