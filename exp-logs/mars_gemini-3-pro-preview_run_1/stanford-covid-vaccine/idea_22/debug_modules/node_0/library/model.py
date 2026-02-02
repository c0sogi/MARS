import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import Config


class LearnableFourierPositionalEncoding(nn.Module):
    """
    Learnable Fourier Positional Encoding (LFPE).
    Encodes signed pairing distances using a bank of learnable frequencies initialized
    with a geometric progression.
    """

    def __init__(self, num_bands=32):
        super().__init__()
        self.num_bands = num_bands
        self.output_dim = num_bands * 2  # sin and cos for each band

        # Initialize frequencies with geometric progression
        # We want to cover wavelengths from roughly 1 to ~200 (sequence length scale)
        # Standard PE uses 10000^(-2i/d). We use a similar logic but make it learnable.
        positions = torch.arange(0, self.num_bands).float()
        # Frequencies decay from 1.0
        freqs = torch.exp(positions * -(math.log(10000.0) / self.num_bands))

        self.freqs = nn.Parameter(freqs)

    def forward(self, distances):
        """
        Args:
            distances (torch.Tensor): Signed pairing distances of shape (B, L).
        Returns:
            torch.Tensor: Encoded features of shape (B, L, output_dim).
        """
        # (B, L, 1) * (num_bands) -> (B, L, num_bands)
        args = distances.unsqueeze(-1) * self.freqs.view(1, 1, -1)

        # Apply sin and cos
        sin_enc = torch.sin(args)
        cos_enc = torch.cos(args)

        # Concatenate: (B, L, 2*num_bands)
        out = torch.cat([sin_enc, cos_enc], dim=-1)
        return out


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Layer.
    Modulates the input features based on the loop context.
    Operation: x_mod = gamma * LayerNorm(x) + beta
    """

    def __init__(self, input_dim, loop_vocab_size, loop_emb_dim=32):
        super().__init__()
        self.input_dim = input_dim

        # Dedicated embedding for modulation parameters
        self.loop_embedding = nn.Embedding(loop_vocab_size, loop_emb_dim)

        # Project loop embedding to gamma (scale) and beta (shift)
        self.projection = nn.Linear(loop_emb_dim, input_dim * 2)

        # Normalization
        self.norm = nn.LayerNorm(input_dim)

        self._init_weights()

    def _init_weights(self):
        # Initialize projection to identity mapping (gamma=1, beta=0)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        # Set bias for gamma part to 1
        with torch.no_grad():
            self.projection.bias[: self.input_dim].fill_(1.0)

    def forward(self, x, loop_indices):
        """
        Args:
            x (torch.Tensor): Input features (B, L, D).
            loop_indices (torch.Tensor): Loop type indices (B, L).
        """
        # Get loop embeddings: (B, L, loop_emb_dim)
        loop_emb = self.loop_embedding(loop_indices)

        # Project to params: (B, L, 2*D)
        params = self.projection(loop_emb)

        gamma, beta = torch.split(params, self.input_dim, dim=-1)

        # Apply normalization
        normed_x = self.norm(x)

        # Modulate
        out = gamma * normed_x + beta
        return out


class ResidualBiGRUBlock(nn.Module):
    """
    Wide-Stream Residual BiGRU Block with FiLM modulation.
    Structure: FiLM -> BiGRU -> Dropout -> Residual Add
    """

    def __init__(self, hidden_dim, loop_vocab_size, dropout=0.1):
        super().__init__()

        # FiLM Layer for structural conditioning
        self.film = FiLMLayer(hidden_dim, loop_vocab_size)

        # Wide-stream BiGRU (Hidden size = Input size / 2 per direction -> Output size = Input size)
        self.bigru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, loop_indices):
        identity = x

        # 1. Structural Modulation
        x = self.film(x, loop_indices)

        # 2. Recurrent Processing
        x, _ = self.bigru(x)

        # 3. Dropout
        x = self.dropout(x)

        # 4. Residual Connection
        out = identity + x
        return out


class ScalarMixtureAggregator(nn.Module):
    """
    Aggregates outputs from multiple layers using a learnable scalar mixture.
    """

    def __init__(self, num_layers):
        super().__init__()
        # Learnable weights for each layer
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, layer_outputs):
        """
        Args:
            layer_outputs (list): List of tensors [(B, L, D), ...].
        Returns:
            torch.Tensor: Weighted sum (B, L, D).
        """
        # Stack: (B, L, D, num_layers)
        stacked = torch.stack(layer_outputs, dim=-1)

        # Softmax weights to ensure they sum to 1
        norm_weights = F.softmax(self.weights, dim=0)

        # Weighted sum
        weighted_sum = torch.sum(stacked * norm_weights, dim=-1)

        return weighted_sum


class RNAModel(nn.Module):
    """
    Structurally Modulated Wide-Stream BiGRU with Learnable Geometric Bias.
    """

    def __init__(self, config=Config):
        super().__init__()

        self.hidden_dim = config.HIDDEN_DIM
        self.num_layers = config.NUM_LAYERS
        self.dropout_rate = config.DROPOUT

        # --- 1. Input Embeddings ---
        # Atomic Nucleotide Embedding
        self.seq_embedding = nn.Embedding(len(config.NUCLEOTIDE_MAP), 32)

        # Loop Embedding (for input concatenation)
        self.loop_embedding_input = nn.Embedding(len(config.LOOP_TYPE_MAP), 32)

        # Learnable Fourier Positional Encoding
        # Using 32 bands creates a 64-dim vector
        self.lfpe = LearnableFourierPositionalEncoding(num_bands=32)
        lfpe_dim = 64

        # Total dimension entering the Stem
        input_dim = 32 + 32 + lfpe_dim  # 128

        # --- 2. Recurrent Stem ---
        # Projects concatenated embeddings to the residual stream width
        self.stem = nn.GRU(
            input_size=input_dim,
            hidden_size=self.hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.stem_dropout = nn.Dropout(self.dropout_rate)

        # --- 3. Backbone ---
        self.blocks = nn.ModuleList(
            [
                ResidualBiGRUBlock(
                    hidden_dim=self.hidden_dim,
                    loop_vocab_size=len(config.LOOP_TYPE_MAP),
                    dropout=self.dropout_rate,
                )
                for _ in range(self.num_layers)
            ]
        )

        # --- 4. Output Head ---
        # Aggregates Stem + N Blocks
        self.aggregator = ScalarMixtureAggregator(num_layers=1 + self.num_layers)

        # Final projection to the 3 scored targets
        num_targets = len(config.TARGET_COLS)
        self.head = nn.Linear(self.hidden_dim, num_targets)

    def forward(self, seq, loop, dist):
        """
        Args:
            seq (torch.Tensor): Sequence indices (B, L).
            loop (torch.Tensor): Loop type indices (B, L).
            dist (torch.Tensor): Signed pairing distances (B, L).
        """
        # Embeddings
        seq_emb = self.seq_embedding(seq)  # (B, L, 32)
        loop_emb = self.loop_embedding_input(loop)  # (B, L, 32)
        dist_emb = self.lfpe(dist)  # (B, L, 64)

        # Concatenate inputs
        x = torch.cat([seq_emb, loop_emb, dist_emb], dim=-1)  # (B, L, 128)

        # Stem Processing
        x, _ = self.stem(x)  # (B, L, hidden_dim)
        x = self.stem_dropout(x)

        # Store outputs for aggregation
        layer_outputs = [x]

        # Pass through Residual Blocks
        for block in self.blocks:
            x = block(x, loop)
            layer_outputs.append(x)

        # Aggregate outputs
        x_agg = self.aggregator(layer_outputs)

        # Project to targets
        logits = self.head(x_agg)  # (B, L, 3)

        return logits
