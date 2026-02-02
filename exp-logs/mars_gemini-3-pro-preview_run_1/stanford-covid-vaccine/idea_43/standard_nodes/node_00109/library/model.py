import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config, get_sinusoidal_encoding


class SinusoidalPositionalEncoding(nn.Module):
    """
    Wrapper for fixed sinusoidal embeddings.
    """

    def __init__(self, num_positions, embedding_dim):
        super().__init__()
        # Retrieve the encoding matrix from the provided utility
        pe = get_sinusoidal_encoding(num_positions, embedding_dim)
        # Register as a non-trainable embedding layer
        self.embedding = nn.Embedding.from_pretrained(pe, freeze=True)

    def forward(self, x):
        return self.embedding(x)


class ResidualBiGRUBlock(nn.Module):
    """
    A single residual block consisting of:
    Pre-LayerNorm -> BiGRU -> Dropout -> Residual Connection.
    Maintains the residual stream width.
    Cite solution_lesson_node_00108: GRU vs. LSTM Efficiency in Low-Data Regimes
    """

    def __init__(self, hidden_dim, dropout_p):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        # BiGRU: hidden_size is halved per direction to sum to hidden_dim
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        residual = x
        out = self.norm(x)
        out, _ = self.gru(out)
        out = self.dropout(out)
        return residual + out


class ScalarMixture(nn.Module):
    """
    Learns a weighted sum of a list of tensors (e.g., layer outputs).
    """

    def __init__(self, num_layers):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, tensors):
        """
        Args:
            tensors: List of tensors, each shape (B, L, D)
        Returns:
            Weighted sum tensor of shape (B, L, D)
        """
        # Stack tensors: (B, L, D, N_layers)
        stacked = torch.stack(tensors, dim=-1)

        # Softmax over the last dimension (layers)
        norm_weights = F.softmax(self.weights, dim=0)

        # Weighted sum
        weighted_sum = torch.sum(stacked * norm_weights, dim=-1)
        return weighted_sum


class RNA_ResBiGRU(nn.Module):
    """
    Main architecture: Stabilized Wide-Stream Residual BiGRU.
    Features:
    - Proportional Embeddings (Seq, Loop, Dist)
    - BiGRU Stem
    - Deep Residual BiGRU Backbone (Width 384)
    - Scalar Mixture Aggregation
    - Shared Output Head
    Cite solution_lesson_node_00108: GRU vs. LSTM Efficiency in Low-Data Regimes
    """

    def __init__(self, config=Config):
        super().__init__()
        self.config = config

        # 1. Embeddings
        self.seq_embed = nn.Embedding(4, config.EMBED_DIM_SEQ)
        self.loop_embed = nn.Embedding(7, config.EMBED_DIM_LOOP)

        # Distance Embedding (Fixed Sinusoidal)
        num_dist_tokens = 2 * config.SEQ_LEN + 1
        self.dist_embed = SinusoidalPositionalEncoding(
            num_dist_tokens, config.EMBED_DIM_DIST
        )
        self.dist_offset = config.SEQ_LEN

        # 2. Recurrent Stem
        # Projects concatenated inputs to residual stream width
        self.stem = nn.GRU(
            input_size=config.TOTAL_INPUT_DIM,
            hidden_size=config.HIDDEN_DIM // 2,
            batch_first=True,
            bidirectional=True,
        )

        # 3. Backbone
        # Stack of Residual BiGRU Blocks
        self.blocks = nn.ModuleList(
            [
                ResidualBiGRUBlock(config.HIDDEN_DIM, config.DROPOUT)
                for _ in range(config.NUM_LAYERS)
            ]
        )

        # 4. Aggregation
        # Mixes output of Stem + 6 Blocks (Total 7)
        self.mixture = ScalarMixture(config.NUM_LAYERS + 1)

        # 5. Output Head
        # Shared projection for the 3 scored targets
        self.head = nn.Linear(config.HIDDEN_DIM, len(config.TARGET_COLS))

    def forward(self, seq, loop, dist):
        # Embed inputs
        x_seq = self.seq_embed(seq)
        x_loop = self.loop_embed(loop)

        # Handle distance offset and embed
        dist_idx = torch.clamp(dist + self.dist_offset, 0, 2 * self.config.SEQ_LEN)
        x_dist = self.dist_embed(dist_idx)

        # Concatenate embeddings
        x = torch.cat([x_seq, x_loop, x_dist], dim=-1)

        # Pass through Stem
        x, _ = self.stem(x)

        # Collect outputs for mixture (starting with Stem output)
        layer_outputs = [x]

        # Pass through Residual Blocks
        for block in self.blocks:
            x = block(x)
            layer_outputs.append(x)

        # Aggregate layers
        x_agg = self.mixture(layer_outputs)

        # Project to targets
        logits = self.head(x_agg)

        return logits
