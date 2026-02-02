import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DilatedResBlock(nn.Module):
    """
    A Residual Block with dilated convolutions to maintain resolution
    while expanding the receptive field.
    """

    def __init__(self, channels, dilation):
        super(DilatedResBlock, self).__init__()
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


class PyramidContextHead(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module.
    Captures multi-scale context using parallel dilated convolutions.
    """

    def __init__(self, in_channels, out_channels, rates):
        super(PyramidContextHead, self).__init__()
        self.stages = nn.ModuleList()

        # 1x1 Conv Branch (Rate 1, effectively)
        self.stages.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        # 3x3 Dilated Conv Branches
        for rate in rates:
            if rate == 1:
                continue  # Already handled by 1x1 or standard 3x3 if desired, but ASPP usually treats rate 1 as 1x1 or 3x3
                # Standard ASPP usually has a 1x1 branch and then 3x3 branches with rates [6, 12, 18].
                # If the config list includes 1, we can treat it as a 3x3 with dilation 1 for texture.

            self.stages.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # Fusion Layer
        # Concatenates all branches
        total_channels = out_channels * len(self.stages)
        self.project = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for stage in self.stages:
            res.append(stage(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class HDNPCA(nn.Module):
    """
    Hierarchical Dilated Network with Pyramid Context Aggregation.
    """

    def __init__(self):
        super(HDNPCA, self).__init__()

        # 1. Learnable 2.5D Projection
        # Compresses Z_DIM (65) -> PROJECTION_DIM
        self.projection = nn.Sequential(
            nn.Conv2d(Config.Z_DIM, Config.PROJECTION_DIM, kernel_size=1, bias=False),
            nn.BatchNorm2d(Config.PROJECTION_DIM),
            nn.ReLU(inplace=True),
        )

        # Adapter to match backbone channels
        self.adapter = nn.Sequential(
            nn.Conv2d(
                Config.PROJECTION_DIM,
                Config.BACKBONE_CHANNELS,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(Config.BACKBONE_CHANNELS),
            nn.ReLU(inplace=True),
        )

        # 2. Sequential Hierarchical Backbone
        # Stack of DilatedResBlocks
        layers = []
        for rate in Config.BACKBONE_DILATIONS:
            layers.append(DilatedResBlock(Config.BACKBONE_CHANNELS, dilation=rate))
        self.backbone = nn.Sequential(*layers)

        # 3. Pyramid Context Aggregation Head (ASPP)
        # Note: Config.ASPP_RATES usually [1, 6, 12, 18].
        # We use BACKBONE_CHANNELS for both in and out of branches to keep width consistent.
        self.aspp = PyramidContextHead(
            Config.BACKBONE_CHANNELS, Config.BACKBONE_CHANNELS, Config.ASPP_RATES
        )

        # 4. Classifier
        self.classifier = nn.Conv2d(Config.BACKBONE_CHANNELS, 1, kernel_size=1)

    def forward(self, x):
        # x shape: (Batch, Z_DIM, H, W)

        # Projection
        x = self.projection(x)  # (B, PROJ_DIM, H, W)

        # Adapter
        x = self.adapter(x)  # (B, BACKBONE_CHANNELS, H, W)

        # Backbone
        x = self.backbone(x)  # (B, BACKBONE_CHANNELS, H, W)

        # Context Head
        x = self.aspp(x)  # (B, BACKBONE_CHANNELS, H, W)

        # Classifier
        logits = self.classifier(x)  # (B, 1, H, W)

        return logits
