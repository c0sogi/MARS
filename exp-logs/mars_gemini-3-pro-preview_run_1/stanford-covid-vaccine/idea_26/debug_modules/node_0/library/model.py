import torch
import torch.nn as nn
import math
from library.config import Config


class SinusoidalSignedPositionalEncoding(nn.Module):
    """
    Encodes signed scalar distances using sinusoidal functions.
    Preserves the sign information (upstream vs downstream) via sin/cos properties.
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Create the division term for the geometric progression of frequencies
        # div_term = 1 / (10000^(2i/d_model))
        # We calculate this once and register as buffer
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len) containing signed distances.
        Returns:
            Tensor of shape (Batch, Seq_Len, d_model)
        """
        # (Batch, Seq, 1) * (d_model/2) -> (Batch, Seq, d_model/2)
        phase = x.unsqueeze(-1) * self.div_term

        pe = torch.zeros(x.size(0), x.size(1), self.d_model, device=x.device)
        pe[..., 0::2] = torch.sin(phase)
        pe[..., 1::2] = torch.cos(phase)
        return pe


class StructuralDropout(nn.Module):
    """
    Randomly drops structural edges (sets pair distance to 0) during training.
    Acts as a regularizer against noisy structure predictions.
    """

    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, pair_dist):
        if not self.training or self.p == 0.0:
            return pair_dist

        # Generate mask: 1 with probability (1-p), 0 with probability p
        # We create a mask of the same shape as pair_dist
        mask = torch.bernoulli(torch.full_like(pair_dist, 1 - self.p))
        return pair_dist * mask


class WideResNetBiGRUBlock(nn.Module):
    """
    A Wide-Stream Residual Block with Pre-LayerNorm and BiGRU.
    Maintains the residual stream width W throughout.
    """

    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size)
        # BiGRU: Hidden size is hidden_size // 2 so that bidirectional output is hidden_size
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-LayerNorm architecture
        residual = x
        out = self.ln(x)

        # BiGRU
        out, _ = self.gru(out)

        # Dropout
        out = self.dropout(out)

        # Residual Connection
        return residual + out


class ScalarMixture(nn.Module):
    """
    Computes a learnable weighted sum of a list of tensors.
    Used to aggregate representations from different depths of the network.
    """

    def __init__(self, n_tensors):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_tensors))

    def forward(self, tensor_list):
        # tensor_list: list of (Batch, Seq, Hidden)
        # Stack: (N, Batch, Seq, Hidden)
        stacked = torch.stack(tensor_list, dim=0)

        # Softmax weights to ensure they sum to 1
        norm_weights = torch.softmax(self.weights, dim=0)

        # Weighted sum: broadcast weights (N, 1, 1, 1)
        weighted_sum = (stacked * norm_weights.view(-1, 1, 1, 1)).sum(dim=0)
        return weighted_sum


class RNAModel(nn.Module):
    """
    Position-Aware Wide-Stream Residual BiGRU with Structural Dropout.
    """

    def __init__(self):
        super().__init__()

        # 1. Embeddings
        # Atomic Sequence Embedding (A, G, C, U)
        self.seq_embed = nn.Embedding(Config.VOCAB_SIZE, Config.EMBED_DIM_SEQ)

        # Predicted Loop Type Embedding
        self.loop_embed = nn.Embedding(Config.LOOP_VOCAB_SIZE, Config.EMBED_DIM_LOOP)

        # Absolute Positional Embedding (0 to 106)
        self.pos_embed = nn.Embedding(Config.SEQ_LENGTH, Config.EMBED_DIM_POS)

        # Signed Pair Distance Embedding (Geometric Encoding)
        # We define a dimension for distance embedding (e.g., 32 to match position/loop)
        self.dist_embed_dim = 32
        self.dist_encoder = SinusoidalSignedPositionalEncoding(self.dist_embed_dim)

        # Structural Dropout (Regularization)
        self.struct_dropout = StructuralDropout(Config.STRUCTURAL_DROPOUT_PROB)

        # Calculate concatenated input dimension
        input_dim = (
            Config.EMBED_DIM_SEQ
            + Config.EMBED_DIM_LOOP
            + Config.EMBED_DIM_POS
            + self.dist_embed_dim
        )

        # 2. Stem
        # Projects concatenated embeddings to the residual stream width (HIDDEN_SIZE)
        # Using a BiGRU for initial contextualization
        self.stem = nn.GRU(
            input_size=input_dim,
            hidden_size=Config.HIDDEN_SIZE // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.stem_dropout = nn.Dropout(Config.DROPOUT)

        # 3. Backbone (Wide-Stream Residual Blocks)
        self.blocks = nn.ModuleList(
            [
                WideResNetBiGRUBlock(Config.HIDDEN_SIZE, Config.DROPOUT)
                for _ in range(Config.NUM_LAYERS)
            ]
        )

        # 4. Output Head
        # Scalar Mixture to aggregate Stem + all Blocks (Total = NUM_LAYERS + 1 tensors)
        self.mixture = ScalarMixture(Config.NUM_LAYERS + 1)

        # Final Projection to targets
        self.head = nn.Linear(Config.HIDDEN_SIZE, Config.NUM_TARGETS)

    def forward(self, batch):
        """
        Args:
            batch (dict): Dictionary containing:
                - sequence: (B, S) Long
                - loop_type: (B, S) Long
                - pair_dist: (B, S) Float
                - position: (B, S) Long
        Returns:
            logits: (B, S, Num_Targets)
        """
        sequence = batch["sequence"]
        loop_type = batch["loop_type"]
        pair_dist = batch["pair_dist"]
        position = batch["position"]

        # A. Structural Dropout (Training only)
        # Randomly zeroes out distance information to force robustness
        pair_dist = self.struct_dropout(pair_dist)

        # B. Embeddings
        emb_seq = self.seq_embed(sequence)  # (B, S, EMBED_DIM_SEQ)
        emb_loop = self.loop_embed(loop_type)  # (B, S, EMBED_DIM_LOOP)
        emb_pos = self.pos_embed(position)  # (B, S, EMBED_DIM_POS)
        emb_dist = self.dist_encoder(pair_dist)  # (B, S, dist_embed_dim)

        # Concatenate all features
        x = torch.cat(
            [emb_seq, emb_loop, emb_pos, emb_dist], dim=-1
        )  # (B, S, input_dim)

        # C. Stem
        # GRU returns (output, h_n). We only need output.
        x_stem, _ = self.stem(x)
        x_stem = self.stem_dropout(x_stem)  # (B, S, HIDDEN_SIZE)

        # D. Backbone
        # We collect outputs for the mixture aggregation
        layer_outputs = [x_stem]
        current_x = x_stem

        for block in self.blocks:
            current_x = block(current_x)
            layer_outputs.append(current_x)

        # E. Aggregation
        # Combine representations from all depths
        aggregated = self.mixture(layer_outputs)  # (B, S, HIDDEN_SIZE)

        # F. Head
        logits = self.head(aggregated)  # (B, S, NUM_TARGETS)

        return logits
