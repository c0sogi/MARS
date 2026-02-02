import torch
import torch.nn as nn
from library.config import Config


class DilatedResidualBlock(nn.Module):
    """
    A residual block with dilated convolutions and Group Normalization.
    Designed to maintain full spatial resolution (no pooling) while increasing
    receptive field via dilation.
    """

    def __init__(self, channels, dilation, groups):
        super(DilatedResidualBlock, self).__init__()

        # First convolution: 3x3 with specific dilation
        # Padding is set to dilation to preserve spatial dimensions (H, W)
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.gn1 = nn.GroupNorm(groups, channels)
        self.relu = nn.ReLU(inplace=True)

        # Second convolution
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
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        out += residual
        out = self.relu(out)

        return out


class DSDN_GN(nn.Module):
    """
    Deep Sequential Dilated Network with Group Normalization (DSDN-GN).

    This architecture is designed to handle the trade-off between model capacity
    and training stability. It uses Group Normalization to allow for deep
    networks even when memory constraints force small batch sizes.
    """

    def __init__(self):
        super(DSDN_GN, self).__init__()

        # 1. Learnable 2.5D Projection
        # Compresses the Z-dimension (65 slices) into a compact feature space.
        # Input: (B, 65, H, W) -> Output: (B, NUM_CHANNELS, H, W)
        self.projection = nn.Sequential(
            nn.Conv2d(Config.Z_DIM, Config.NUM_CHANNELS, kernel_size=1, bias=False),
            nn.GroupNorm(Config.GN_NUM_GROUPS, Config.NUM_CHANNELS),
            nn.ReLU(inplace=True),
        )

        # 2. Deep Sequential Dilated Backbone
        # Constructs a sequence of residual blocks with increasing dilation rates.
        layers = []
        dilation_rates = Config.get_dilation_rates()

        for dilation in dilation_rates:
            layers.append(
                DilatedResidualBlock(
                    channels=Config.NUM_CHANNELS,
                    dilation=dilation,
                    groups=Config.GN_NUM_GROUPS,
                )
            )

        self.backbone = nn.Sequential(*layers)

        # 3. Classification Head
        # Maps features to binary logits.
        # Output: (B, 1, H, W)
        self.head = nn.Conv2d(Config.NUM_CHANNELS, 1, kernel_size=1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Applies Kaiming initialization to convolutions and constant init to GroupNorm.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
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
