import torch
import torch.nn as nn
from library.config import Config


class DilatedResidualBlock(nn.Module):
    """
    A Residual Block with Dilated Convolutions to expand receptive field
    without losing spatial resolution.
    Structure: Conv(d) -> BN -> ReLU -> Conv(d) -> BN -> Add -> ReLU
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


class DCDN(nn.Module):
    """
    Densely Connected Dilated Network (DCDN).

    Features:
    1. Learnable 2.5D Projection (Depth -> Channels).
    2. Sequential Dilated Backbone (No pooling).
    3. Global Feature Aggregation (Dense skip connections).
    """

    def __init__(self):
        super(DCDN, self).__init__()

        # 1. Learnable 2.5D Projection
        # Compresses the 65 depth slices into a compact feature representation
        self.projection = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,
            out_channels=Config.PROJECTION_DIM,
            kernel_size=1,
            bias=False,
        )
        self.proj_bn = nn.BatchNorm2d(Config.PROJECTION_DIM)
        self.proj_relu = nn.ReLU(inplace=True)

        # 2. Sequential Dilated Backbone
        # Hierarchical dilation rates defined in Config (e.g., 1, 2, 4, 8)
        self.backbone = nn.ModuleList()

        # Verify consistent width
        if Config.PROJECTION_DIM != Config.BACKBONE_WIDTH:
            raise ValueError(
                "Projection dim must match backbone width for this implementation."
            )

        for dilation in Config.DILATION_RATES:
            block = DilatedResidualBlock(Config.BACKBONE_WIDTH, dilation)
            self.backbone.append(block)

        # 3. Output Head
        # Calculates input channels based on dense aggregation of all block outputs
        total_backbone_channels = len(Config.DILATION_RATES) * Config.BACKBONE_WIDTH

        self.head = nn.Conv2d(
            in_channels=total_backbone_channels, out_channels=1, kernel_size=1
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input volume of shape (B, Z, H, W).
                              Z is treated as input channels.
        Returns:
            torch.Tensor: Logits of shape (B, 1, H, W).
        """
        # Projection Stage
        x = self.projection(x)
        x = self.proj_bn(x)
        x = self.proj_relu(x)

        # Backbone Stage with Feature Collection
        features = []
        for block in self.backbone:
            x = block(x)
            features.append(x)

        # Global Feature Aggregation
        # Concatenate features from all dilated blocks (Hypercolumn)
        # Shape: (B, num_blocks * width, H, W)
        aggregated_features = torch.cat(features, dim=1)

        # Output Stage
        logits = self.head(aggregated_features)

        return logits
