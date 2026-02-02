import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import Config


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Encodes signed scalar distances using fixed sinusoidal functions.
    Preserves the sign of the distance to distinguish upstream/downstream relationships.
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Create constant 'div_term' for frequencies: 1 / 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len) containing signed float distances.
        Returns:
            Tensor of shape (Batch, Seq_Len, d_model)
        """
        # Unsqueeze to get (Batch, Seq_Len, 1)
        x = x.unsqueeze(-1)

        # Compute arguments: x * frequencies
        # Broadcasting: (B, L, 1) * (D/2,) -> (B, L, D/2)
        args = x * self.div_term

        # Initialize output tensor
        pe = torch.zeros(x.shape[0], x.shape[1], self.d_model, device=x.device)

        # Apply Sin to even indices, Cos to odd indices
        pe[:, :, 0::2] = torch.sin(args)
        pe[:, :, 1::2] = torch.cos(args)

        return pe


class GatedFusionStem(nn.Module):
    """
    Fuses input embeddings using a local convolution and a Gated Linear Unit (GLU).
    This allows for non-linear interaction of features before temporal propagation.
    """

    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        # GLU halves the dimension, so Conv1d output must be 2 * hidden_dim
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim * 2,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, x):
        # x: (Batch, Seq_Len, input_dim)
        x = self.norm(x)

        # Conv1d expects (Batch, Channels, Seq_Len)
        x = x.permute(0, 2, 1)
        x = self.conv(x)

        # GLU over channel dimension (dim=1)
        # Input: (B, 2*H, L) -> Output: (B, H, L)
        x = F.glu(x, dim=1)

        # Permute back to (Batch, Seq_Len, Hidden_Dim)
        x = x.permute(0, 2, 1)
        return x


class ResidualBiGRUBlock(nn.Module):
    """
    A Wide-Stream Residual Block using Pre-LayerNorm and BiGRU.
    Maintains the full residual width throughout.
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.bigru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,  # Bidirectional, so output is hidden_dim
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-LayerNorm Residual Connection
        residual = x

        out = self.norm(x)
        out, _ = self.bigru(out)
        out = self.dropout(out)

        return residual + out


class ScalarMixture(nn.Module):
    """
    Computes a learnable weighted sum of a list of tensors.
    """

    def __init__(self, n_layers):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_layers))

    def forward(self, tensors):
        """
        Args:
            tensors: List of tensors, all same shape (B, L, D)
        """
        # Normalize weights using Softmax
        norm_weights = F.softmax(self.weights, dim=0)

        # Weighted sum
        out = torch.zeros_like(tensors[0])
        for i, t in enumerate(tensors):
            out = out + norm_weights[i] * t

        return out


class RNAModel(nn.Module):
    """
    Gated-Fusion Wide-Stream Residual BiGRU Model.
    """

    def __init__(self, config=Config):
        super().__init__()

        # 1. Input Embeddings
        self.seq_embed = nn.Embedding(config.VOCAB_SIZE, config.EMBED_DIM)
        self.loop_embed = nn.Embedding(config.LOOP_VOCAB_SIZE, config.EMBED_DIM)
        self.dist_embed = SinusoidalPositionalEmbedding(config.EMBED_DIM)

        # 2. Gated Fusion Stem
        # Concatenation of 3 embeddings -> Input Dim
        fusion_in_dim = config.EMBED_DIM * 3
        self.fusion_stem = GatedFusionStem(
            input_dim=fusion_in_dim,
            hidden_dim=config.HIDDEN_DIM,
            kernel_size=config.FUSION_KERNEL_SIZE,
        )

        # 3. Recurrent Contextualization (Stem RNN)
        # Projects stream to residual width and provides initial temporal context
        self.stem_rnn = nn.GRU(
            input_size=config.HIDDEN_DIM,
            hidden_size=config.HIDDEN_DIM // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # 4. Backbone: Residual Blocks
        self.blocks = nn.ModuleList(
            [
                ResidualBiGRUBlock(config.HIDDEN_DIM, dropout=config.DROPOUT)
                for _ in range(config.N_LAYERS)
            ]
        )

        # 5. Scalar Mixture Aggregation
        # Aggregates outputs from Stem RNN + All Blocks
        self.mixture = ScalarMixture(n_layers=config.N_LAYERS + 1)

        # 6. Output Head
        self.head = nn.Linear(config.HIDDEN_DIM, config.NUM_TARGETS)

    def forward(self, sequence, loop_type, pair_dist, **kwargs):
        """
        Forward pass of the model.

        Args:
            sequence: (Batch, Seq_Len) LongTensor
            loop_type: (Batch, Seq_Len) LongTensor
            pair_dist: (Batch, Seq_Len) FloatTensor
            **kwargs: Ignored (e.g., ids)

        Returns:
            logits: (Batch, Seq_Len, Num_Targets)
        """
        # Embed Inputs
        e_seq = self.seq_embed(sequence)  # (B, L, 128)
        e_loop = self.loop_embed(loop_type)  # (B, L, 128)
        e_dist = self.dist_embed(pair_dist)  # (B, L, 128)

        # Concatenate
        x = torch.cat([e_seq, e_loop, e_dist], dim=-1)  # (B, L, 384)

        # Gated Fusion
        x = self.fusion_stem(x)  # (B, L, 384)

        # Recurrent Contextualization
        x, _ = self.stem_rnn(x)  # (B, L, 384)

        # Collect outputs for mixture (Start with Stem output)
        layer_outputs = [x]

        # Pass through Residual Blocks
        for block in self.blocks:
            x = block(x)
            layer_outputs.append(x)

        # Aggregate Features
        x_agg = self.mixture(layer_outputs)

        # Prediction Head
        logits = self.head(x_agg)

        return logits
