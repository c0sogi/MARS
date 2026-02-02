import torch
import torch.nn as nn
import torch.nn.init as init
from library.config import Config


class ResidualBlock(nn.Module):
    """
    Standard Residual Block with Zero-Gamma Initialization support.
    Structure: Conv -> BN -> ReLU -> Conv -> BN -> Add Input
    """

    def __init__(self, channels, kernel_size=3, padding=1, use_zero_gamma=True):
        super(ResidualBlock, self).__init__()

        # First convolution block
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        # Second convolution block
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

        # Initialize weights
        self._initialize_weights(use_zero_gamma)

    def _initialize_weights(self, use_zero_gamma):
        # Kaiming initialization for convolutions
        init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")

        # BN1 initialization (Standard: gamma=1, beta=0)
        init.constant_(self.bn1.weight, 1)
        init.constant_(self.bn1.bias, 0)

        # BN2 initialization
        # Zero-Gamma: gamma=0, beta=0. This makes the block an identity mapping at init.
        if use_zero_gamma:
            init.constant_(self.bn2.weight, 0)
        else:
            init.constant_(self.bn2.weight, 1)
        init.constant_(self.bn2.bias, 0)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        return out


class ZIResDnCNN(nn.Module):
    """
    Zero-Initialized Deep Residual Denoising Network.
    Uses global residual learning (predicts noise) and a deep stack of residual blocks.
    """

    def __init__(
        self,
        num_blocks=Config.NUM_BLOCKS,
        num_channels=Config.NUM_CHANNELS,
        kernel_size=Config.KERNEL_SIZE,
        padding=Config.PADDING,
        use_zero_gamma=Config.USE_ZERO_GAMMA,
    ):
        super(ZIResDnCNN, self).__init__()

        # Input Layer: 1 Channel (Grayscale) -> Num Channels
        # Bias is True here as there is no BN following it immediately
        self.input_conv = nn.Conv2d(
            1, num_channels, kernel_size, padding=padding, bias=True
        )
        self.input_relu = nn.ReLU(inplace=True)

        # Backbone: Stack of Residual Blocks
        layers = []
        for _ in range(num_blocks):
            layers.append(
                ResidualBlock(num_channels, kernel_size, padding, use_zero_gamma)
            )
        self.backbone = nn.Sequential(*layers)

        # Output Layer: Num Channels -> 1 Channel (Noise Map)
        self.output_conv = nn.Conv2d(
            num_channels, 1, kernel_size, padding=padding, bias=True
        )

        # Initialize Input/Output layers
        self._initialize_outer_layers()

    def _initialize_outer_layers(self):
        # Input Conv
        init.kaiming_normal_(
            self.input_conv.weight, mode="fan_out", nonlinearity="relu"
        )
        if self.input_conv.bias is not None:
            init.constant_(self.input_conv.bias, 0)

        # Output Conv
        init.kaiming_normal_(
            self.output_conv.weight, mode="fan_out", nonlinearity="linear"
        )
        if self.output_conv.bias is not None:
            init.constant_(self.output_conv.bias, 0)

    def forward(self, x):
        # Feature Extraction
        out = self.input_conv(x)
        out = self.input_relu(out)

        # Residual Blocks
        out = self.backbone(out)

        # Noise Prediction
        noise = self.output_conv(out)

        # Global Residual Learning: Clean Image = Input - Noise
        clean_image = x - noise

        return clean_image
