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


class ParallelContextHead(nn.Module):
    """
    Aggregates features from multiple receptive fields in parallel at the end of the network.
    """

    def __init__(self, in_channels, dilation_rates):
        super().__init__()
        self.branches = nn.ModuleList()

        for rate in dilation_rates:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True),
                )
            )

        concat_channels = in_channels * len(dilation_rates)

        # Fusion layer to combine parallel context
        self.fusion = nn.Sequential(
            nn.Conv2d(concat_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        out = torch.cat(branch_outputs, dim=1)
        out = self.fusion(out)
        return out


class ESDN_PCH(nn.Module):
    """
    Extended Sequential Dilated Network with Parallel Context Head.

    Architecture:
    1. Projection: Z_DIM -> BACKBONE_CHANNELS (1x1 Conv)
    2. Backbone: Sequential Dilated Residual Blocks (r=1,2,4,8,16,32)
    3. Head: Parallel Context Module (r=1,6,12,18)
    4. Classifier: 1x1 Conv -> 1 channel
    """

    def __init__(self):
        super().__init__()

        # 1. Projection Layer
        self.projection = ProjectionLayer(Config.Z_DIM, Config.BACKBONE_CHANNELS)

        # 2. Extended Sequential Dilated Backbone
        backbone_layers = []
        for rate in Config.BACKBONE_DILATION_RATES:
            backbone_layers.append(
                DilatedResidualBlock(Config.BACKBONE_CHANNELS, dilation=rate)
            )
        self.backbone = nn.Sequential(*backbone_layers)

        # 3. Parallel Context Head
        self.head = ParallelContextHead(
            Config.BACKBONE_CHANNELS, Config.HEAD_DILATION_RATES
        )

        # 4. Final Classifier
        self.classifier = nn.Conv2d(Config.BACKBONE_CHANNELS, 1, kernel_size=1)

    def forward(self, x):
        # Input x: (Batch, 65, H, W)

        # Project Z-dimension
        x = self.projection(x)

        # Extract features with large receptive field
        x = self.backbone(x)

        # Refine with multi-scale parallel context
        x = self.head(x)

        # Classify
        logits = self.classifier(x)

        return logits
