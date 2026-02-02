import torch
import torch.nn as nn
from library.config import Config


class ResidualBlock(nn.Module):
    """
    Standard Residual Block with dilated convolutions.
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


class InkDetectorFCN(nn.Module):
    """
    Sequential Dilated FCN.

    Architecture:
    1. Learnable 2.5D Projection (Cite solution_lesson_node_00002)
    2. Sequential Backbone of Dilated Residual Blocks (Cite solution_lesson_node_00003, solution_lesson_node_00015)
    3. Final classification head
    """

    def __init__(self):
        super().__init__()

        # 1. Learnable 2.5D Projection
        self.projection = nn.Sequential(
            nn.Conv2d(Config.Z_DIM, Config.MODEL_CHANNELS, kernel_size=1, bias=False),
            nn.BatchNorm2d(Config.MODEL_CHANNELS),
            nn.ReLU(inplace=True),
        )

        # 2. Backbone: Sequential Dilated Residual Blocks
        layers = []
        for rate in Config.DILATION_RATES:
            layers.append(ResidualBlock(channels=Config.MODEL_CHANNELS, dilation=rate))
        self.backbone = nn.Sequential(*layers)

        # 3. Classification Head
        self.classifier = nn.Conv2d(Config.MODEL_CHANNELS, 1, kernel_size=1)

    def forward(self, x):
        x = self.projection(x)
        x = self.backbone(x)
        logits = self.classifier(x)
        return logits
