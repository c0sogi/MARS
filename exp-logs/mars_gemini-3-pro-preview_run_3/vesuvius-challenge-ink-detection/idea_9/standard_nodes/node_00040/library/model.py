import torch
import torch.nn as nn
from library.config import Config


class ProjectionLayer(nn.Module):
    """
    Compresses the 3D input volume (treated as channels) into a 2D feature map
    using a learnable 1x1 convolution.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x shape: (Batch, Z_DIM, H, W)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DilatedResidualBlock(nn.Module):
    """
    A standard Residual Block with dilated convolutions to expand receptive field
    without losing resolution.
    """

    def __init__(self, channels, dilation):
        super().__init__()
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


class DilatedFCN(nn.Module):
    """
    Sequential Dilated Fully Convolutional Network.

    Simplified architecture focusing on sequential hierarchical context aggregation
    (Cite solution_lesson_node_00015) with increased capacity (Cite solution_lesson_node_00036).

    Architecture:
    1. Projection: Z_DIM -> BACKBONE_CHANNELS (1x1 Conv)
    2. Backbone: Sequential Dilated Residual Blocks
    3. Classifier: 1x1 Conv -> 1 channel
    """

    def __init__(self):
        super().__init__()

        # 1. Projection Layer
        self.projection = ProjectionLayer(Config.Z_DIM, Config.BACKBONE_CHANNELS)

        # 2. Sequential Dilated Backbone
        backbone_layers = []
        for rate in Config.BACKBONE_DILATION_RATES:
            backbone_layers.append(
                DilatedResidualBlock(Config.BACKBONE_CHANNELS, dilation=rate)
            )
        self.backbone = nn.Sequential(*backbone_layers)

        # 3. Final Classifier
        self.classifier = nn.Conv2d(Config.BACKBONE_CHANNELS, 1, kernel_size=1)

    def forward(self, x):
        # Input x: (Batch, 65, H, W)

        # Project Z-dimension
        x = self.projection(x)

        # Extract features with large receptive field
        x = self.backbone(x)

        # Classify
        logits = self.classifier(x)

        return logits
