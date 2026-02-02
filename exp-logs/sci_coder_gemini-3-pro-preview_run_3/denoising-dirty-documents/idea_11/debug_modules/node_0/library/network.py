import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention for Efficient Mobile Network Design.
    Factorizes channel attention into two parallel 1D feature encoding processes.
    """

    def __init__(self, inp, reduction=16):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # C x H x 1
        x_h = self.pool_h(x)
        # C x 1 x W
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # Concatenate along spatial dimension -> C x (H+W) x 1
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split back
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # Expand to original channel depth
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w
        return out


class ResidualBlock(nn.Module):
    """
    Residual Block with Coordinate Attention and Zero-Gamma Initialization.
    Structure: Conv -> BN -> ReLU -> Conv -> BN -> CA -> Add
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        # Reduction ratio 16 for 64 channels -> 4 bottleneck channels
        self.ca = CoordinateAttention(channels, reduction=16)

        # Zero-Gamma Initialization
        # Initialize the last BN weight to 0 so the block is identity at start
        nn.init.constant_(self.bn2.weight, 0)
        nn.init.constant_(self.bn2.bias, 0)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out)

        out += residual
        return out


class CAResDnCNN(nn.Module):
    """
    Coordinate Attention Stabilized Deep Residual Network (CA-ResDnCNN).
    Predicts the noise residual from the input image.
    """

    def __init__(
        self,
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        num_features=Config.NUM_FEATURES,
        num_blocks=Config.NUM_BLOCKS,
    ):
        super(CAResDnCNN, self).__init__()

        # Head: Transform input to feature space
        self.head = nn.Conv2d(
            in_channels, num_features, kernel_size=3, padding=1, bias=True
        )

        # Body: Deep stack of Residual Blocks
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(num_features))
        self.body = nn.Sequential(*layers)

        # Tail: Transform features to noise residual output
        self.tail = nn.Conv2d(
            num_features, out_channels, kernel_size=3, padding=1, bias=True
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # Note: BatchNorm weights are handled in ResidualBlock for Zero-Gamma
            # or default to 1/0 for others. We don't override them here globally
            # to preserve the Zero-Gamma logic.

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Input tensor (B, C, H, W) - Noisy Image
        Returns:
            noise: Predicted noise residual (B, C, H, W)
        """
        out = self.head(x)
        out = self.body(out)
        noise = self.tail(out)
        return noise
