import torch
import torch.nn as nn
from library.config import Config


class DilatedResidualBlock(nn.Module):
    """
    A residual block with dilated convolutions and Batch Normalization.
    Maintains full spatial resolution.
    """

    def __init__(self, channels, dilation):
        super(DilatedResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class ResidualFCN(nn.Module):
    """
    Residual Fully Convolutional Network (Residual FCN).

    A lean, shallow architecture optimized for single-sample training regimes.
    Uses Batch Normalization and Dilated Convolutions.
    Cite solution_lesson_node_00004: Lean Spatial Backbones
    """

    def __init__(self):
        super(ResidualFCN, self).__init__()

        # 1. Learnable 2.5D Projection
        # Cite solution_lesson_node_00002: 2.5D Volumetric Projection
        self.projection = nn.Sequential(
            nn.Conv2d(Config.Z_DIM, Config.NUM_CHANNELS, kernel_size=1, bias=False),
            nn.BatchNorm2d(Config.NUM_CHANNELS),
            nn.ReLU(inplace=True),
        )

        # 2. Dilated Backbone
        layers = []
        dilation_rates = Config.get_dilation_rates()

        for dilation in dilation_rates:
            layers.append(
                DilatedResidualBlock(
                    channels=Config.NUM_CHANNELS,
                    dilation=dilation,
                )
            )

        self.backbone = nn.Sequential(*layers)

        # 3. Classification Head
        self.head = nn.Conv2d(Config.NUM_CHANNELS, 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input volume tensor of shape (Batch, 65, H, W).

        Returns:
            torch.Tensor: Logits of shape (Batch, 1, H, W).
        """
        # Project 3D volume to 2D features
        x = self.projection(x)

        # Apply deep dilated backbone
        x = self.backbone(x)

        # Generate logits
        x = self.head(x)

        return x
