import torch
import torch.nn as nn
from library.config import Config


class LayerNormChannels(nn.Module):
    """
    Applies LayerNorm along the channel dimension of a (N, C, L) tensor.
    Permutes dimensions to (N, L, C) for normalization and back.
    """

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: (N, C, L)
        x = x.transpose(1, 2)  # (N, L, C)
        x = self.norm(x)
        x = x.transpose(1, 2)  # (N, C, L)
        return x


class DecoupledConvBlock(nn.Module):
    """
    A robust decoupled convolutional block with pre-activation structure.
    Structure: LayerNorm -> SiLU -> Dilated Conv (k=3) -> LayerNorm -> SiLU -> Pointwise Conv (k=1) -> Dropout

    This block is designed to be used within a DenseNet-style architecture where
    input channels may vary (grow) but output channels (growth rate) are fixed.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()

        # 1. Pre-activation & Dilated Convolution
        # Normalizes the input features before processing
        self.ln1 = LayerNormChannels(in_channels)
        self.act1 = nn.SiLU()

        # Padding to maintain sequence length: (k-1) * d // 2
        padding = (kernel_size - 1) * dilation // 2

        # Standard Full-Rank Convolution for spatial mixing
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )

        # 2. Pre-activation & Pointwise Convolution
        # Normalizes the intermediate features
        self.ln2 = LayerNormChannels(out_channels)
        self.act2 = nn.SiLU()

        # Pointwise Convolution for channel mixing/projection
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1)

        # 3. Regularization
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (N, in_channels, L)

        # First sub-block (Spatial)
        out = self.ln1(x)
        out = self.act1(out)
        out = self.conv1(out)

        # Second sub-block (Channel)
        out = self.ln2(out)
        out = self.act2(out)
        out = self.conv2(out)

        out = self.drop(out)

        return out


class DenseTCN(nn.Module):
    """
    A Temporal Convolutional Network with Dense Connections.

    Iterates through a list of dilation rates. At each step, a DecoupledConvBlock
    is applied to the concatenation of the original input and all previous block outputs.

    This architecture ensures maximum information flow and gradient propagation.
    """

    def __init__(self, in_channels, growth_rate, dilations, kernel_size, dropout):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.growth_rate = growth_rate
        self.dilations = dilations

        current_in_channels = in_channels

        for d in dilations:
            block = DecoupledConvBlock(
                in_channels=current_in_channels,
                out_channels=growth_rate,
                kernel_size=kernel_size,
                dilation=d,
                dropout=dropout,
            )
            self.blocks.append(block)

            # In a DenseNet, the input to the next layer grows by the growth_rate
            current_in_channels += growth_rate

        # The final output channel count (original input + all block outputs)
        self.out_channels = current_in_channels

    def forward(self, x):
        # x: (N, in_channels, L)

        # List to store all feature maps, starting with the input
        features = [x]

        for block in self.blocks:
            # Dense Connection: Concatenate all previous features along channel dim
            inp = torch.cat(features, dim=1)

            # Compute new features
            new_features = block(inp)

            # Add new features to the list
            features.append(new_features)

        # Return the full dense representation
        # Shape: (N, in_channels + num_layers * growth_rate, L)
        return torch.cat(features, dim=1)
