import torch
import torch.nn as nn
from library.config import Config


class ResidualBlock(nn.Module):
    """
    Standard Residual Block with Additive Skip Connection.
    Cite solution_lesson_node_00029: Prefer additive feature refinement over multiplicative gating.
    """

    def __init__(self, channels, dilation, groups):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.gn1 = nn.GroupNorm(groups, channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.gn2 = nn.GroupNorm(groups, channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out = self.act(out + residual)
        return out


class SGDN(nn.Module):
    """
    Residual Fully Convolutional Network (ResidualFCN).
    Cite solution_lesson_node_00003: Residual FCN utilizing dilated convolutions.
    """

    def __init__(self):
        super(SGDN, self).__init__()

        # Load architecture hyperparameters from Config
        self.z_dim = Config.Z_DIM
        self.base_channels = Config.BASE_CHANNELS
        self.dilation_rates = Config.DILATION_RATES
        self.groups = Config.GROUP_NORM_GROUPS

        # 1. Learnable 2.5D Projection (Input Stage)
        self.projection = nn.Conv2d(
            in_channels=self.z_dim,
            out_channels=self.base_channels,
            kernel_size=1,
            bias=False,
        )

        # 2. Residual Backbone
        layers = []
        for d in self.dilation_rates:
            layers.append(
                ResidualBlock(
                    channels=self.base_channels, dilation=d, groups=self.groups
                )
            )
        self.backbone = nn.Sequential(*layers)

        # 3. Classification Head
        self.head = nn.Conv2d(
            in_channels=self.base_channels, out_channels=1, kernel_size=1, bias=True
        )

    def forward(self, x):
        # Project Z-dimension to feature channels
        x = self.projection(x)

        # Pass through the backbone
        x = self.backbone(x)

        # Generate logits
        logits = self.head(x)

        return logits
