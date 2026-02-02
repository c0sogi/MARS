import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import Config


class SinusoidalPositionalEncoding(nn.Module):
    """
    Encodes scalar values (pairing distances) into a high-dimensional vector
    using fixed sinusoidal functions. Preserves sign information.

    Implements the 'Explicit Geometric Encoding' with 'Fixed Inductive Bias'.
    """

    def __init__(self, d_model):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.d_model = d_model

        # Compute the frequencies for the geometric progression
        # div_term = 1 / 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        """
        Args:
            x: (Batch, Seq_Len) tensor of float distances.
        Returns:
            (Batch, Seq_Len, d_model) tensor.
        """
        # x is (B, L) -> (B, L, 1)
        # div_term is (d_model/2) -> (1, 1, d_model/2)
        # Result: (B, L, d_model/2)
        scaled_x = x.unsqueeze(-1) * self.div_term.view(1, 1, -1)

        # Create output tensor (B, L, d_model)
        pe = torch.zeros(x.size(0), x.size(1), self.d_model, device=x.device)

        # Apply sin to even indices, cos to odd indices
        pe[:, :, 0::2] = torch.sin(scaled_x)
        pe[:, :, 1::2] = torch.cos(scaled_x)

        return pe


class GroupedBiGRUBlock(nn.Module):
    """
    Implements the Split-Transform-Merge topology for the Wide-Stream backbone.

    1. Pre-Norm
    2. Split input into 'num_groups' chunks
    3. Transform each chunk with an independent BiGRU
    4. Merge (Concatenate) and Mix (Linear)
    5. Residual Connection
    """

    def __init__(self, hidden_dim, num_groups, dropout):
        super(GroupedBiGRUBlock, self).__init__()
        self.num_groups = num_groups
        self.hidden_dim = hidden_dim

        if hidden_dim % num_groups != 0:
            raise ValueError(
                f"Hidden dim {hidden_dim} must be divisible by num_groups {num_groups}"
            )

        self.group_dim = hidden_dim // num_groups

        # Pre-Norm
        self.norm = nn.LayerNorm(hidden_dim)

        # Parallel Recurrence (Transform)
        # Each GRU processes a slice of the hidden state.
        # Bidirectional: Hidden size per direction = group_dim // 2
        # Output size per group = group_dim
        self.grus = nn.ModuleList(
            [
                nn.GRU(
                    input_size=self.group_dim,
                    hidden_size=self.group_dim // 2,
                    bidirectional=True,
                    batch_first=True,
                )
                for _ in range(num_groups)
            ]
        )

        # Mixing Projection (Merge)
        self.linear = nn.Linear(hidden_dim, hidden_dim)

        # Regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Seq, Hidden)
        residual = x

        # Pre-Norm
        x = self.norm(x)

        # Split
        chunks = x.chunk(self.num_groups, dim=-1)

        # Transform
        gru_outs = []
        for i, gru in enumerate(self.grus):
            # gru output: (Batch, Seq, group_dim)
            out, _ = gru(chunks[i])
            gru_outs.append(out)

        # Merge
        concatenated = torch.cat(gru_outs, dim=-1)

        # Mix
        mixed = self.linear(concatenated)

        # Residual
        out = residual + self.dropout(mixed)

        return out


class ScalarMixture(nn.Module):
    """
    Computes a learnable weighted sum of layer outputs (Global Static Aggregation).
    Uses Softmax normalization on weights.
    """

    def __init__(self, num_layers):
        super(ScalarMixture, self).__init__()
        # Initialize weights to 0 (resulting in equal uniform distribution after softmax)
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, layer_outputs):
        """
        Args:
            layer_outputs: List of tensors, each (Batch, Seq, Hidden)
        Returns:
            Tensor of shape (Batch, Seq, Hidden)
        """
        # Stack: (Num_Layers, Batch, Seq, Hidden)
        stacked = torch.stack(layer_outputs, dim=0)

        # Normalize weights
        norm_weights = F.softmax(self.weights, dim=0)

        # Broadcast weights for multiplication: (Num_Layers, 1, 1, 1)
        norm_weights = norm_weights.view(-1, 1, 1, 1)

        # Weighted Sum
        weighted_sum = torch.sum(norm_weights * stacked, dim=0)

        return weighted_sum


class RNAModel(nn.Module):
    """
    Cardinality-Scaled Wide-Stream Residual BiGRU Model.
    """

    def __init__(self, config=Config):
        super(RNAModel, self).__init__()

        # --- 1. Embeddings ---
        self.embed_dim = config.EMBED_DIM

        # Atomic Sequence Embedding (A, G, C, U)
        self.seq_embed = nn.Embedding(config.VOCAB_SIZE, self.embed_dim)

        # Loop Type Embedding (Structure Context)
        self.loop_embed = nn.Embedding(config.LOOP_TYPES, self.embed_dim)

        # Distance Encoding (Geometric Bias)
        self.dist_embed = SinusoidalPositionalEncoding(self.embed_dim)

        # Input Dimension to Stem (Concatenation of 3 channels)
        input_dim = self.embed_dim * 3

        # --- 2. Recurrent Stem ---
        self.hidden_dim = config.HIDDEN_DIM
        self.stem = nn.GRU(
            input_size=input_dim,
            hidden_size=self.hidden_dim // 2,
            bidirectional=True,
            batch_first=True,
        )

        # --- 3. Backbone (Cardinality-Scaled) ---
        self.num_layers = config.NUM_LAYERS
        self.blocks = nn.ModuleList(
            [
                GroupedBiGRUBlock(
                    hidden_dim=self.hidden_dim,
                    num_groups=config.NUM_GROUPS,
                    dropout=config.DROPOUT,
                )
                for _ in range(self.num_layers)
            ]
        )

        # --- 4. Aggregation ---
        # We aggregate outputs from the Stem + all Blocks
        self.mixture = ScalarMixture(self.num_layers + 1)

        # --- 5. Output Head ---
        # Shared projection to targets
        self.head = nn.Linear(self.hidden_dim, config.NUM_TARGETS)

    def forward(self, sequence, loop_type, pair_dist):
        """
        Args:
            sequence: (Batch, Seq) LongTensor
            loop_type: (Batch, Seq) LongTensor
            pair_dist: (Batch, Seq) FloatTensor
        Returns:
            logits: (Batch, Seq, Num_Targets)
        """
        # 1. Embed Inputs
        emb_seq = self.seq_embed(sequence)  # (B, L, 128)
        emb_loop = self.loop_embed(loop_type)  # (B, L, 128)
        emb_dist = self.dist_embed(pair_dist)  # (B, L, 128)

        # Concatenate
        x = torch.cat([emb_seq, emb_loop, emb_dist], dim=-1)  # (B, L, 384)

        # 2. Stem Processing
        x, _ = self.stem(x)  # (B, L, 512)

        # Store for aggregation
        layer_outputs = [x]

        # 3. Backbone Processing
        for block in self.blocks:
            x = block(x)
            layer_outputs.append(x)

        # 4. Aggregation
        x_agg = self.mixture(layer_outputs)

        # 5. Prediction
        logits = self.head(x_agg)  # (B, L, 3)

        return logits
