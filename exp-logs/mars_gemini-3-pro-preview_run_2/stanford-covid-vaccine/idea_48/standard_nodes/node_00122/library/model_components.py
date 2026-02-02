import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormChannel(nn.Module):
    """
    Applies Layer Normalization to a channel-first tensor (N, C, L).
    PyTorch's LayerNorm expects (N, *, C), so we transpose, normalize, and transpose back.
    """

    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        # x: (N, C, L) -> (N, L, C)
        x = x.transpose(1, 2)
        x = self.norm(x)
        # (N, L, C) -> (N, C, L)
        return x.transpose(1, 2)


class InputEmbeddingStem(nn.Module):
    """
    Projects sparse One-Hot encoded inputs into a dense embedding space.
    This is critical before applying Pre-Activation LayerNorm to avoid destroying
    the discrete signal structure.

    Args:
        in_channels (int): Total number of input feature channels.
        out_channels (int): Dimension of the dense embedding (e.g., 64).
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Standard Convolution (Kernel 1) acts as a learnable linear projection per position
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (N, in_channels, L) -> (N, out_channels, L)
        return self.conv(x)


class DenseDilatedBlock(nn.Module):
    """
    A Single-Layer Dilated Block utilizing a Pre-Activation structure.
    Designed for use in a DenseNet-style architecture where inputs are concatenations
    of prior feature maps.

    Structure:
    LayerNorm -> SiLU -> Dilated Conv (k=3) -> LayerNorm -> SiLU -> Pointwise Conv (k=1) -> Dropout
    """

    def __init__(
        self, in_channels, growth_rate, kernel_size=3, dilation=1, dropout=0.1
    ):
        super().__init__()

        # 1. First Pre-Activation Unit: LN -> SiLU -> Dilated Conv
        self.norm1 = LayerNormChannel(in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(
            in_channels,
            growth_rate,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
        )

        # 2. Second Pre-Activation Unit: LN -> SiLU -> Pointwise Conv
        # Note: conv1 outputs 'growth_rate' channels
        self.norm2 = LayerNormChannel(growth_rate)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv1d(growth_rate, growth_rate, kernel_size=1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (N, in_channels, L)

        # Unit 1
        out = self.norm1(x)
        out = self.act1(out)
        out = self.conv1(out)

        # Unit 2
        out = self.norm2(out)
        out = self.act2(out)
        out = self.conv2(out)

        out = self.dropout(out)

        # Output: (N, growth_rate, L)
        return out


class PureFeedbackModule(nn.Module):
    """
    Encapsulates the feedback processing logic.
    Takes recycled predictions, embeds them, and processes them through a
    Lightweight Dense TCN.

    Structure:
    1. Embedding (Conv1d)
    2. Stack of DenseDilatedBlocks (with dense concatenation)
    3. Final Projection to Feedback Embedding Dimension
    """

    def __init__(
        self, in_channels, hidden_dim, growth_rate, out_channels, dilations, dropout=0.1
    ):
        super().__init__()

        # Initial projection of predictions (e.g., 5 channels -> hidden_dim)
        self.embedding = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)

        self.blocks = nn.ModuleList()
        current_dim = hidden_dim

        # Create dense blocks
        for d in dilations:
            block = DenseDilatedBlock(
                in_channels=current_dim,
                growth_rate=growth_rate,
                kernel_size=3,
                dilation=d,
                dropout=dropout,
            )
            self.blocks.append(block)
            # In a DenseNet, the input to the next layer grows by growth_rate
            current_dim += growth_rate

        # Final projection to the desired feedback embedding size (E_fb)
        self.final_proj = nn.Conv1d(current_dim, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (N, 5, L) - Recycled predictions

        # 1. Embed
        features = self.embedding(x)  # (N, hidden_dim, L)

        # 2. Dense Processing
        # We maintain 'features' as the accumulated dense state
        for block in self.blocks:
            new_features = block(features)
            # Concatenate along channel dimension (Dense Connection)
            features = torch.cat([features, new_features], dim=1)

        # 3. Project to E_fb
        # features now has shape (N, hidden_dim + num_blocks*growth_rate, L)
        out = self.final_proj(features)

        # Output: (N, out_channels, L)
        return out
