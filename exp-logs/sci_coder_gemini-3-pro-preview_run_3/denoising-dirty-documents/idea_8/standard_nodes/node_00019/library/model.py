import torch
import torch.nn as nn
from library.config import Config


class ResidualBlock(nn.Module):
    """
    Residual Block for ResDnCNN.
    Structure: Conv -> BN -> ReLU -> Conv -> BN -> Element-wise Add.
    Maintains spatial resolution (no pooling).
    """

    def __init__(self, num_features):
        """
        Args:
            num_features (int): Number of input/output feature channels.
        """
        super(ResidualBlock, self).__init__()

        # First Conv-BN-ReLU
        self.conv1 = nn.Conv2d(
            in_channels=num_features,
            out_channels=num_features,
            kernel_size=Config.KERNEL_SIZE,
            padding=Config.PADDING,
            bias=False,  # Bias is redundant with BatchNorm
        )
        self.bn1 = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)

        # Second Conv-BN
        self.conv2 = nn.Conv2d(
            in_channels=num_features,
            out_channels=num_features,
            kernel_size=Config.KERNEL_SIZE,
            padding=Config.PADDING,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(num_features)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Local Skip Connection
        out += identity

        # Note: No final ReLU after addition to allow full range of values in residual path
        return out


class ResDnCNN(nn.Module):
    """
    Deep Residual Denoising Network (ResDnCNN).
    Constructed from a deep stack of ResidualBlocks to predict the noise residual.
    """

    def __init__(self):
        super(ResDnCNN, self).__init__()

        in_channels = Config.IN_CHANNELS
        num_features = Config.NUM_FEATURES
        num_blocks = Config.NUM_RES_BLOCKS
        kernel_size = Config.KERNEL_SIZE
        padding = Config.PADDING

        # Head: Initial Feature Extraction (Conv + ReLU)
        # No BN in the first layer usually
        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels, num_features, kernel_size, padding=padding, bias=True
            ),
            nn.ReLU(inplace=True),
        )

        # Body: Stack of Residual Blocks
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(num_features))
        self.body = nn.Sequential(*blocks)

        # Tail: Reconstruction Layer (Conv)
        # Maps features back to the noise residual space (1 channel)
        self.tail = nn.Conv2d(
            num_features, in_channels, kernel_size, padding=padding, bias=True
        )

        self._initialize_weights()

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input noisy image tensor [B, C, H, W].

        Returns:
            torch.Tensor: Predicted noise residual [B, C, H, W].
        """
        out = self.head(x)
        out = self.body(out)
        noise = self.tail(out)
        return noise

    def _initialize_weights(self):
        """
        Initialize weights using Kaiming Normal for Convs and Constant for BN.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
