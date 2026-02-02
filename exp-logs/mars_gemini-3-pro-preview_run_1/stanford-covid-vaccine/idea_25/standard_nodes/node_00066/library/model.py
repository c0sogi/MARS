import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import Config


class SinusoidalDistanceEmbedding(nn.Module):
    """
    Encodes scalar signed distances using sinusoidal functions.
    Preserves sign information via phase.
    """

    def __init__(self, dim, min_timescale=1.0, max_timescale=100.0):
        super().__init__()
        self.dim = dim
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale

        # Create fixed frequencies
        num_timescales = self.dim // 2
        log_timescales = torch.linspace(
            math.log(self.min_timescale), math.log(self.max_timescale), num_timescales
        )
        inv_timescales = torch.exp(-log_timescales)
        self.register_buffer("inv_timescales", inv_timescales)

    def forward(self, x):
        # x shape: (batch, seq_len) containing signed distances
        # Output shape: (batch, seq_len, dim)

        # Expand dims for broadcasting: (batch, seq_len, 1) * (num_timescales)
        scaled_x = x.unsqueeze(-1) * self.inv_timescales.view(1, 1, -1)

        # Sin and Cos
        sin_x = torch.sin(scaled_x)
        cos_x = torch.cos(scaled_x)

        # Concatenate: (batch, seq_len, dim)
        # If dim is odd, we might need padding, but we usually choose even dims
        emb = torch.cat([sin_x, cos_x], dim=-1)

        if self.dim % 2 == 1:
            # Pad with zero if dim is odd (rare case)
            emb = F.pad(emb, (0, 1))

        return emb


class StochasticDepth(nn.Module):
    """
    Implements Stochastic Depth (LayerDrop) with inverted scaling.
    During training, drops the input with probability (1 - survival_prob)
    and scales by (1 / survival_prob).
    During eval, acts as identity.
    """

    def __init__(self, survival_prob=1.0):
        super().__init__()
        self.survival_prob = survival_prob

    def forward(self, x):
        if not self.training or self.survival_prob == 1.0:
            return x

        # Bernoulli mask
        # Shape: (batch, 1, 1) to drop entire samples or (batch, ...) for per-element?
        # Standard LayerDrop drops the entire residual branch for a sample in the batch.
        batch_size = x.shape[0]
        noise_tensor = torch.empty(
            batch_size, 1, 1, device=x.device, dtype=x.dtype
        ).bernoulli_(self.survival_prob)

        # Scale and mask
        return x * noise_tensor / self.survival_prob


class ResidualBiGRUBlock(nn.Module):
    """
    A Wide-Stream Residual Block using BiGRU.
    Structure: Input -> LN -> BiGRU -> StochasticDepth -> + -> Output
    """

    def __init__(self, hidden_dim, dropout=0.1, survival_prob=1.0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        # BiGRU: Hidden size is hidden_dim // 2 so output is hidden_dim
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.stochastic_depth = StochasticDepth(survival_prob)

    def forward(self, x):
        residual = x

        # Pre-LayerNorm
        out = self.layer_norm(x)

        # Transformation
        out, _ = self.gru(out)
        out = self.dropout(out)

        # Stochastic Depth on the residual branch
        out = self.stochastic_depth(out)

        # Residual Connection
        return residual + out


class ScalarMixture(nn.Module):
    """
    Computes a learnable weighted sum of a list of tensors.
    """

    def __init__(self, num_layers):
        super().__init__()
        # Initialize weights to be equal (0.0 before softmax makes them equal)
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, layer_outputs):
        # layer_outputs: List of tensors, each (batch, seq_len, dim)
        # Stack: (batch, seq_len, dim, num_layers)
        stacked = torch.stack(layer_outputs, dim=-1)

        # Softmax weights
        norm_weights = F.softmax(self.weights, dim=0)

        # Weighted sum
        # (batch, seq_len, dim, num_layers) * (num_layers) -> sum over last dim
        weighted_sum = torch.sum(stacked * norm_weights, dim=-1)

        return weighted_sum


class DynamicDepthWideStreamBiGRU(nn.Module):
    """
    Main architecture:
    1. Embeddings (Seq, Loop, Dist)
    2. Recurrent Stem (BiGRU)
    3. Deep Stack of Residual BiGRU Blocks with Stochastic Depth
    4. Scalar Mixture Aggregation
    5. Shared Output Head
    """

    def __init__(self):
        super().__init__()

        # 1. Embeddings
        self.seq_embedding = nn.Embedding(Config.VOCAB_SIZE_SEQ, Config.EMBED_DIM)
        self.loop_embedding = nn.Embedding(Config.VOCAB_SIZE_LOOP, Config.EMBED_DIM)
        self.dist_embedding = SinusoidalDistanceEmbedding(Config.EMBED_DIM)

        # Input projection dimension: 3 * EMBED_DIM
        input_dim = 3 * Config.EMBED_DIM

        # 2. Recurrent Stem
        # Projects to HIDDEN_DIM (512)
        self.stem = nn.GRU(
            input_size=input_dim,
            hidden_size=Config.HIDDEN_DIM // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.stem_dropout = nn.Dropout(Config.DROPOUT)

        # 3. Backbone: Dynamic-Depth Blocks
        self.blocks = nn.ModuleList()

        # Calculate linear decay for survival probability
        start_prob = Config.LAYER_DROP_SURVIVAL_START
        end_prob = Config.LAYER_DROP_SURVIVAL_END
        num_layers = Config.NUM_LAYERS

        for i in range(num_layers):
            # Linear decay: p_i = start - (start - end) * i / (num_layers - 1)
            # If num_layers is 1, prob is start.
            if num_layers > 1:
                p_l = start_prob - (start_prob - end_prob) * i / (num_layers - 1)
            else:
                p_l = start_prob

            block = ResidualBiGRUBlock(
                hidden_dim=Config.HIDDEN_DIM, dropout=Config.DROPOUT, survival_prob=p_l
            )
            self.blocks.append(block)

        # 4. Scalar Mixture
        # We aggregate outputs from Stem + all Blocks
        self.mixture = ScalarMixture(num_layers=num_layers + 1)

        # 5. Output Head
        self.head = nn.Linear(Config.HIDDEN_DIM, Config.NUM_TARGETS)

    def forward(self, seq, loop, dist, mask=None):
        # Embeddings
        emb_seq = self.seq_embedding(seq)  # (B, L, E)
        emb_loop = self.loop_embedding(loop)  # (B, L, E)
        emb_dist = self.dist_embedding(dist)  # (B, L, E)

        # Concatenate
        x = torch.cat([emb_seq, emb_loop, emb_dist], dim=-1)

        # Stem
        x, _ = self.stem(x)
        x = self.stem_dropout(x)

        # Collect outputs for mixture
        layer_outputs = [x]

        # Pass through blocks
        for block in self.blocks:
            x = block(x)
            layer_outputs.append(x)

        # Aggregate
        x_aggregated = self.mixture(layer_outputs)

        # Output Head
        logits = self.head(x_aggregated)  # (B, L, 3)

        return logits
