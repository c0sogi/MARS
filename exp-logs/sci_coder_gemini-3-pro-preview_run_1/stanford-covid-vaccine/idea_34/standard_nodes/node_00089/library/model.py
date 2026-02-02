import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Encodes scalar distances using fixed sinusoidal functions.
    Preserves the sign of the distance to distinguish direction (upstream vs downstream).
    Strictly uses fixed inductive bias over learnable embeddings for geometry.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # We use half the dimension for sin and half for cos to create dim features
        half_dim = dim // 2

        # Compute the frequency divisors
        # Formula: exp(i * -log(10000) / half_dim) corresponds to 10000^(-2i/dim)
        div_term = torch.exp(
            torch.arange(0, half_dim).float() * (-math.log(10000.0) / half_dim)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        """
        Args:
            x: (Batch, Seq_Len) tensor of signed float distances.
        Returns:
            (Batch, Seq_Len, Dim) tensor of positional encodings.
        """
        # Expand to (Batch, Seq_Len, 1) for broadcasting
        x = x.unsqueeze(-1)

        # Compute arguments: (Batch, Seq_Len, Half_Dim)
        args = x * self.div_term

        # Compute Sin and Cos
        pe_sin = torch.sin(args)
        pe_cos = torch.cos(args)

        # Concatenate to get full dimension: (Batch, Seq_Len, Dim)
        # Note: Assumes dim is even (Config.emb_dim=128 is even)
        pe = torch.cat([pe_sin, pe_cos], dim=-1)
        return pe


class ResidualBiGRUBlock(nn.Module):
    """
    Wide-Stream Residual Block.
    Maintains the full residual stream width throughout the block.
    Structure: Pre-LN -> BiGRU -> Dropout -> Residual Add.
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)

        # BiGRU: Hidden size is hidden_dim // 2 per direction to sum to hidden_dim
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            bidirectional=True,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        # Pre-LayerNorm
        out = self.ln(x)

        # BiGRU Processing
        out, _ = self.gru(out)

        # Dropout
        out = self.dropout(out)

        # Residual Connection
        return residual + out


class ScalarMixture(nn.Module):
    """
    Global Static Aggregation.
    Computes a learnable weighted sum of outputs from different layers.
    """

    def __init__(self, n_layers):
        super().__init__()
        self.n_layers = n_layers
        # Learnable weights for each layer
        self.weights = nn.Parameter(torch.zeros(n_layers))

    def forward(self, tensors):
        """
        Args:
            tensors: List of tensors, each of shape (Batch, Seq_Len, Hidden_Dim).
        Returns:
            (Batch, Seq_Len, Hidden_Dim) weighted sum tensor.
        """
        # Stack tensors: (Batch, Seq_Len, Hidden_Dim, N_Layers)
        stacked = torch.stack(tensors, dim=-1)

        # Compute softmax weights for stability: (N_Layers,)
        probs = F.softmax(self.weights, dim=0)

        # Weighted sum across the last dimension
        # Broadcasting probs to (1, 1, 1, N_Layers)
        weighted_sum = torch.sum(stacked * probs, dim=-1)

        return weighted_sum


class BondAwareModel(nn.Module):
    """
    Bond-Aware Wide-Stream Residual BiGRU Model.

    Architecture:
    1. Embeddings: Atomic Sequence, Loop Type, Bond Type (Soft Feature), Distance (Fixed).
    2. Stem: BiGRU projecting concatenated embeddings to residual width (384).
    3. Backbone: 6x ResidualBiGRUBlocks (Pre-LN).
    4. Aggregation: Scalar Mixture of Stem + 6 Blocks.
    5. Head: Shared Linear Projection to 3 targets.
    """

    def __init__(self):
        super().__init__()

        # --------------------------
        # 1. Embeddings
        # --------------------------
        self.seq_emb = nn.Embedding(Config.vocab_size_seq, Config.emb_dim)
        self.loop_emb = nn.Embedding(Config.vocab_size_loop, Config.emb_dim)
        self.bond_emb = nn.Embedding(Config.vocab_size_bond, Config.emb_dim)
        self.dist_emb = SinusoidalPositionalEmbedding(Config.emb_dim)

        # --------------------------
        # 2. Recurrent Stem
        # --------------------------
        # Input: Concatenation of 4 feature channels
        input_dim = 4 * Config.emb_dim

        # Projects input to model width (Config.hidden_dim)
        self.stem = nn.GRU(
            input_size=input_dim,
            hidden_size=Config.hidden_dim // 2,
            bidirectional=True,
            batch_first=True,
        )

        # --------------------------
        # 3. Backbone
        # --------------------------
        self.blocks = nn.ModuleList(
            [
                ResidualBiGRUBlock(Config.hidden_dim, Config.dropout)
                for _ in range(Config.n_layers)
            ]
        )

        # --------------------------
        # 4. Aggregation
        # --------------------------
        # Aggregates outputs from Stem + all Blocks
        self.mixture = ScalarMixture(Config.n_layers + 1)

        # --------------------------
        # 5. Output Head
        # --------------------------
        # Projects to 3 targets: reactivity, deg_Mg_pH10, deg_Mg_50C
        self.head = nn.Linear(Config.hidden_dim, 3)

    def forward(self, seq, loop, bond, dist):
        """
        Args:
            seq: (Batch, Seq_Len) LongTensor
            loop: (Batch, Seq_Len) LongTensor
            bond: (Batch, Seq_Len) LongTensor
            dist: (Batch, Seq_Len) FloatTensor
        Returns:
            logits: (Batch, Seq_Len, 3) FloatTensor
        """
        # 1. Embed Features
        e_seq = self.seq_emb(seq)  # (B, L, emb_dim)
        e_loop = self.loop_emb(loop)  # (B, L, emb_dim)
        e_bond = self.bond_emb(bond)  # (B, L, emb_dim)
        e_dist = self.dist_emb(dist)  # (B, L, emb_dim)

        # 2. Concatenate
        x = torch.cat([e_seq, e_loop, e_bond, e_dist], dim=-1)  # (B, L, 4*emb_dim)

        # 3. Stem
        x, _ = self.stem(x)  # (B, L, hidden_dim)

        # Collect outputs for mixture (starting with Stem output)
        layer_outputs = [x]

        # 4. Backbone Blocks
        for block in self.blocks:
            x = block(x)
            layer_outputs.append(x)

        # 5. Aggregation
        x_agg = self.mixture(layer_outputs)  # (B, L, hidden_dim)

        # 6. Head
        logits = self.head(x_agg)  # (B, L, 3)

        return logits
