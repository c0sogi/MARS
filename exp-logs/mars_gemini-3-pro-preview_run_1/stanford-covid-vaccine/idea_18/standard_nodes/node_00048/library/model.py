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


class ResidualBiGRUBlock(nn.Module):
    """
    A residual block containing a BiGRU layer with Pre-LayerNorm.
    """

    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.bigru = nn.GRU(
            input_size=dim,
            hidden_size=dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.ln(x)
        out, _ = self.bigru(out)
        out = self.dropout(out)
        return residual + out


class RNAResidualBiGRU(nn.Module):
    """
    Deep Residual BiGRU Architecture.

    Features:
    - Atomic Nucleotide & Loop Embeddings
    - Signed Sinusoidal Distance Encoding
    - Recurrent Stem (BiGRU Projection)
    - Stack of Residual BiGRU Blocks (Pre-LN)
    - Linear Output Head
    """

    def __init__(self):
        super().__init__()

        # Configuration
        self.hidden_dim = Config.HIDDEN_DIM
        self.num_layers = Config.NUM_LAYERS
        self.dropout_rate = Config.DROPOUT

        # Internal Embedding Dimensions
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
        self.stem = nn.GRU(
            input_size=stem_input_dim,
            hidden_size=self.hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.stem_dropout = nn.Dropout(self.dropout_rate)

        # 3. Backbone: Residual BiGRU Blocks
        self.blocks = nn.ModuleList(
            [
                ResidualBiGRUBlock(
                    dim=self.hidden_dim,
                    dropout=self.dropout_rate,
                )
                for _ in range(self.num_layers)
            ]
        )

        # 4. Output Head
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, 3),
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
