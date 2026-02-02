import torch
import torch.nn as nn
from library.config import Config


class ResidualBlock(nn.Module):
    """
    A linear residual block structure: Conv -> BN -> ReLU -> Conv -> BN.
    Includes the critical Zero-Gamma initialization for the second BN layer
    to facilitate deep residual learning by starting as an identity mapping.
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=Config.KERNEL_SIZE,
            padding=Config.PADDING,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=Config.KERNEL_SIZE,
            padding=Config.PADDING,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize Conv1
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")

        # Initialize BN1 (Standard: gamma=1, beta=0)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)

        # Initialize Conv2
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")

        # Initialize BN2 (Zero-Gamma: gamma=0, beta=0)
        # This forces the block to output 0 + input = input at initialization,
        # preventing signal explosion in deep networks.
        if Config.ZERO_GAMMA_INIT:
            nn.init.constant_(self.bn2.weight, 0)
        else:
            nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        return out


class EZ_ResDnCNN(nn.Module):
    """
    Ensemble of Zero-Initialized Deep Residual Networks (EZ-ResDnCNN).

    This model is designed to predict the noise residual from a noisy input image.
    It utilizes a deep stack of residual blocks with zero-gamma initialization
    to allow training of very deep networks (20+ blocks) without convergence issues.
    """

    def __init__(self):
        super(EZ_ResDnCNN, self).__init__()

        in_channels = Config.IN_CHANNELS
        out_channels = Config.OUT_CHANNELS
        num_features = Config.NUM_FEATURES
        num_blocks = Config.NUM_RES_BLOCKS
        kernel_size = Config.KERNEL_SIZE
        padding = Config.PADDING

        # Head: Feature extraction (Input -> Features)
        # Using bias=True here as there is no BN following this layer
        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels,
                num_features,
                kernel_size=kernel_size,
                padding=padding,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )

        # Body: Deep denoising via residual learning
        self.body = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_blocks)]
        )

        # Tail: Reconstruction (Features -> Noise Residual)
        # Using bias=True here as there is no BN following this layer
        self.tail = nn.Conv2d(
            num_features,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

        self._initialize_head_tail()

    def _initialize_head_tail(self):
        # Initialize Head
        for m in self.head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Initialize Tail
        # We assume a linear activation for the final noise prediction
        nn.init.kaiming_normal_(self.tail.weight, mode="fan_out", nonlinearity="linear")
        if self.tail.bias is not None:
            nn.init.constant_(self.tail.bias, 0)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Noisy input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Predicted noise residual of shape (B, C, H, W).
        """
        features = self.head(x)
        features = self.body(features)
        noise = self.tail(features)
        return noise
