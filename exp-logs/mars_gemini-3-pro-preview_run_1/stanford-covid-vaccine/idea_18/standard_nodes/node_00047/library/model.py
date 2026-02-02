import torch
import torch.nn as nn
import numpy as np
from library.config import Config


class SinusoidalDistanceEmbedding(nn.Module):
    """
    Encodes signed scalar distances into high-dimensional vectors using
    sinusoidal functions, preserving sign information.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        if dim % 2 != 0:
            raise ValueError(f"Sinusoidal embedding dimension must be even, got {dim}")

        # Precompute frequency terms: 10000^(2i/dim)
        # We use a standard geometric progression of frequencies
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
        self.register_buffer("div_term", div_term)

    def forward(self, dists):
        """
        Args:
            dists: (Batch, SeqLen) tensor of signed float distances.
        Returns:
            (Batch, SeqLen, Dim) tensor.
        """
        # dists: (B, L) -> (B, L, 1)
        dists = dists.unsqueeze(-1)

        # div_term: (dim/2,) -> (1, 1, dim/2)
        div_term = self.div_term.view(1, 1, -1)

        # Compute phase
        phase = dists * div_term

        # Compute sin and cos components
        pe_sin = torch.sin(phase)
        pe_cos = torch.cos(phase)

        # Concatenate: (B, L, dim)
        return torch.cat([pe_sin, pe_cos], dim=-1)


class PointwiseMLP(nn.Module):
    """
    A Pointwise Feed-Forward Network applied to each position independently.
    Structure: Linear -> GELU -> Dropout -> Linear -> Dropout
    """

    def __init__(self, dim, expansion_factor, dropout=0.0):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class WideStreamBlock(nn.Module):
    """
    A residual block containing a Wide-Stream BiGRU sub-layer and a Pointwise MLP sub-layer.
    Uses Pre-LayerNorm configuration.
    """

    def __init__(self, dim, expansion_factor, dropout=0.0):
        super().__init__()

        # 1. Wide-Stream BiGRU Sub-layer
        # The BiGRU maintains the residual stream width.
        # Hidden size is dim // 2 per direction => Output is dim.
        self.ln1 = nn.LayerNorm(dim)
        self.bigru = nn.GRU(
            input_size=dim,
            hidden_size=dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        # 2. Pointwise MLP Sub-layer
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = PointwiseMLP(dim, expansion_factor, dropout)

    def forward(self, x):
        # Sub-layer 1: BiGRU
        residual = x
        out = self.ln1(x)
        out, _ = self.bigru(out)
        out = self.dropout1(out)
        x = residual + out

        # Sub-layer 2: MLP
        residual = x
        out = self.ln2(x)
        out = self.mlp(out)
        x = residual + out

        return x


class InterleavedBiGRU(nn.Module):
    """
    Interleaved Wide-Stream BiGRU-MLP Architecture.

    Features:
    - Atomic Nucleotide & Loop Embeddings
    - Signed Sinusoidal Distance Encoding
    - Recurrent Stem (BiGRU Projection)
    - Stack of Interleaved Wide-Stream Blocks (BiGRU + MLP)
    - MLP Output Head
    """

    def __init__(self):
        super().__init__()

        # Configuration
        self.hidden_dim = Config.HIDDEN_DIM
        self.num_layers = Config.NUM_LAYERS
        self.dropout_rate = Config.DROPOUT
        self.expansion = Config.MLP_EXPANSION_FACTOR

        # Internal Embedding Dimensions
        # We use a moderate dimension for features before the stem projection
        self.emb_dim_seq = 64
        self.emb_dim_loop = 64
        self.emb_dim_dist = 64

        # 1. Input Embeddings
        self.seq_embedding = nn.Embedding(Config.VOCAB_SIZE_SEQ, self.emb_dim_seq)
        self.loop_embedding = nn.Embedding(Config.VOCAB_SIZE_LOOP, self.emb_dim_loop)

        if Config.USE_DISTANCE_ENCODING:
            self.dist_embedding = SinusoidalDistanceEmbedding(self.emb_dim_dist)
            stem_input_dim = self.emb_dim_seq + self.emb_dim_loop + self.emb_dim_dist
        else:
            self.dist_embedding = None
            stem_input_dim = self.emb_dim_seq + self.emb_dim_loop

        # 2. Recurrent Stem
        # Projects concatenated embeddings to hidden_dim using a single BiGRU layer.
        # This contextualizes features temporally before the deep backbone.
        self.stem = nn.GRU(
            input_size=stem_input_dim,
            hidden_size=self.hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.stem_dropout = nn.Dropout(self.dropout_rate)

        # 3. Backbone: Interleaved Wide-Stream Blocks
        self.blocks = nn.ModuleList(
            [
                WideStreamBlock(
                    dim=self.hidden_dim,
                    expansion_factor=self.expansion,
                    dropout=self.dropout_rate,
                )
                for _ in range(self.num_layers)
            ]
        )

        # 4. Output Head
        # Projects final hidden states to the 3 scored targets.
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, 3),  # reactivity, deg_Mg_pH10, deg_Mg_50C
        )

        # Weight Initialization
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights for better convergence.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
            # GRU weights are usually initialized reasonably by default.

    def forward(self, sequence, loop_type, pair_dist):
        """
        Forward pass of the model.

        Args:
            sequence: (Batch, SeqLen) LongTensor of nucleotide indices.
            loop_type: (Batch, SeqLen) LongTensor of loop type indices.
            pair_dist: (Batch, SeqLen) FloatTensor of signed pairing distances.

        Returns:
            (Batch, SeqLen, 3) FloatTensor of predictions.
        """
        # 1. Embeddings
        emb_seq = self.seq_embedding(sequence)  # (B, L, 64)
        emb_loop = self.loop_embedding(loop_type)  # (B, L, 64)

        features = [emb_seq, emb_loop]

        if self.dist_embedding is not None:
            emb_dist = self.dist_embedding(pair_dist)  # (B, L, 64)
            features.append(emb_dist)

        # Concatenate features along the channel dimension
        x = torch.cat(features, dim=-1)  # (B, L, 192)

        # 2. Stem
        x, _ = self.stem(x)  # (B, L, 384)
        x = self.stem_dropout(x)

        # 3. Backbone
        for block in self.blocks:
            x = block(x)

        # 4. Head
        out = self.head(x)  # (B, L, 3)

        return out
