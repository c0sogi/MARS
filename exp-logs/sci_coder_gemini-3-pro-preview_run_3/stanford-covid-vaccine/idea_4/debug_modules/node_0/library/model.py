import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class FeedForwardModule(nn.Module):
    """
    Conformer Feed Forward Module.
    Structure: LN -> Linear -> SiLU -> Dropout -> Linear -> Dropout
    Uses a 0.5 scale factor on the residual output (Macaron-Net style).
    """

    def __init__(self, dim, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * expansion_factor)
        self.activation = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * expansion_factor, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Seq_Len, Dim)
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return residual + 0.5 * x


class MultiHeadSelfAttentionModule(nn.Module):
    """
    Conformer Multi-Head Self Attention Module.
    Structure: LN -> MHSA -> Dropout -> Residual
    """

    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Seq_Len, Dim)
        residual = x
        x = self.layer_norm(x)
        # Self-attention with no mask (full context)
        x, _ = self.attention(x, x, x)
        x = self.dropout(x)
        return residual + x


class ConformerConvModule(nn.Module):
    """
    Conformer Convolution Module.
    Structure: LN -> Pointwise Conv -> GLU -> Depthwise Conv -> BN -> Swish -> Pointwise Conv -> Dropout
    """

    def __init__(self, dim, kernel_size, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)

        # Pointwise convolution (expansion for GLU)
        self.pointwise_conv1 = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.glu = nn.GLU(dim=1)

        # Depthwise convolution
        # Padding ensures output length matches input length
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=padding, groups=dim  # Depthwise
        )

        self.batch_norm = nn.BatchNorm1d(dim)
        self.activation = nn.SiLU()

        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Seq_Len, Dim)
        residual = x
        x = self.layer_norm(x)

        # Transpose for Conv1d: (Batch, Dim, Seq_Len)
        x = x.transpose(1, 2)

        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        # Transpose back: (Batch, Seq_Len, Dim)
        x = x.transpose(1, 2)

        return residual + x


class ConformerBlock(nn.Module):
    """
    Single Conformer Block.
    Composition:
    1. FeedForward (Half-step)
    2. MultiHeadSelfAttention
    3. Convolution
    4. FeedForward (Half-step)
    5. LayerNorm
    """

    def __init__(self, dim, num_heads, kernel_size, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForwardModule(dim, expansion_factor=4, dropout=dropout)
        self.mhsa = MultiHeadSelfAttentionModule(dim, num_heads, dropout=dropout)
        self.conv = ConformerConvModule(dim, kernel_size, dropout=dropout)
        self.ff2 = FeedForwardModule(dim, expansion_factor=4, dropout=dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.ff1(x)
        x = self.mhsa(x)
        x = self.conv(x)
        x = self.ff2(x)
        x = self.layer_norm(x)
        return x


class RNAConformer(nn.Module):
    """
    Main Conformer-based RNA Degradation Predictor.
    """

    def __init__(self):
        super().__init__()

        # Load hyperparameters from Config
        dim = Config.DIM_MODEL
        num_heads = Config.NUM_HEADS
        num_layers = Config.NUM_LAYERS
        kernel_size = Config.CONV_KERNEL_SIZE
        dropout = Config.DROPOUT
        input_channels = Config.INPUT_CHANNELS
        output_channels = Config.OUTPUT_CHANNELS
        seq_len = Config.SEQ_LENGTH

        # 1. Input Projection
        # Projects one-hot features (14 channels) to model dimension
        self.embedding = nn.Linear(input_channels, dim)

        # 2. Positional Encoding
        # Learnable positional embeddings
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, dim))

        # 3. Conformer Encoder
        self.layers = nn.ModuleList(
            [
                ConformerBlock(dim, num_heads, kernel_size, dropout)
                for _ in range(num_layers)
            ]
        )

        # 4. Output Head
        self.head = nn.Linear(dim, output_channels)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize positional encoding with small values
        nn.init.normal_(self.pos_encoding, mean=0.0, std=0.02)

        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Seq_Len, 14)
        Returns:
            torch.Tensor: Predictions of shape (Batch, Seq_Len, 5)
        """
        # Project input
        x = self.embedding(x)  # (B, L, Dim)

        # Add positional encoding
        x = x + self.pos_encoding

        # Pass through Conformer blocks
        for layer in self.layers:
            x = layer(x)

        # Project to output targets
        x = self.head(x)  # (B, L, 5)

        return x
