import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from library.config import Config


class ResidualBlock(nn.Module):
    """
    A Residual Block consisting of two 3x3 convolutions with Batch Normalization and ReLU.
    Includes a skip connection (identity mapping) to facilitate gradient flow and
    mitigate the vanishing gradient problem in deep networks.
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        # First convolution block
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second convolution block
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        # If input and output channels differ, use 1x1 conv to match dimensions
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class ResUNet(nn.Module):
    """
    Residual U-Net Architecture.
    Combines the symmetric U-Net encoder-decoder structure with Residual Blocks
    to enable deeper architectures and better feature learning for denoising.
    """

    def __init__(
        self,
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        features=Config.FEATURES,
    ):
        super(ResUNet, self).__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Encoder (Contracting Path) ---
        # Construct downsampling layers based on feature depths
        current_in_channels = in_channels
        for feature in features[:-1]:
            self.downs.append(ResidualBlock(current_in_channels, feature))
            current_in_channels = feature

        # --- Bottleneck ---
        # The bridge between encoder and decoder
        self.bottleneck = ResidualBlock(features[-2], features[-1])

        # --- Decoder (Expanding Path) ---
        # Construct upsampling layers in reverse feature order
        for feature in reversed(features[:-1]):
            # Up-convolution (Upsample + Conv) doubles spatial dim, halves channels
            # Prefer Upsampling to avoid checkerboard artifacts (Cite solution_lesson_node_00009)
            self.ups.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.Conv2d(
                        feature * 2,
                        feature,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                )
            )
            # Residual Block processes the concatenated features
            # Input channels: feature*2 (skip connection + upsampled), Output: feature
            self.ups.append(ResidualBlock(feature * 2, feature))

        # --- Final Output Layer ---
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder pass
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Reverse skip connections for easy access during decoding
        skip_connections = skip_connections[::-1]

        # Decoder pass
        for i in range(0, len(self.ups), 2):
            # Apply Up-convolution
            x = self.ups[i](x)

            # Retrieve corresponding skip connection
            skip_connection = skip_connections[i // 2]

            # Handle potential dimension mismatch (e.g. if input size was odd)
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            # Concatenate along channel dimension
            concat_skip = torch.cat((skip_connection, x), dim=1)

            # Apply Residual Block
            x = self.ups[i + 1](concat_skip)

        return self.final_conv(x)
