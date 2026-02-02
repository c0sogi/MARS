import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block.
    Sequence: Conv2d -> BN -> ReLU -> MaxPool2d.
    Removed Spatial Dropout (Cite Lesson 69) and SE Blocks (Cite Lesson 31).
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class SimpleCNN(nn.Module):
    """
    Simple 4-Stage Convolutional Network.
    Optimized based on Lessons:
    - No SE Blocks (Lesson 31).
    - No Spatial Dropout (Lesson 69).
    - Standard Global Max Pooling on final layer only (Lesson 35).
    - Default Initialization (Lesson 45).
    - Shallow Dense Head (Lesson 40).
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Hyperparameters
        channels = Config.CHANNEL_SIZES  # Expected: [64, 128, 128, 128]
        head_dropout = Config.HEAD_DROPOUT_RATE

        # Backbone Stages
        self.block1 = ConvBlock(Config.IN_CHANNELS, channels[0])
        self.block2 = ConvBlock(channels[0], channels[1])
        self.block3 = ConvBlock(channels[1], channels[2])
        self.block4 = ConvBlock(channels[2], channels[3])

        # Global Pooling (Max Pooling for sparse, high-intensity signals)
        self.global_pool = nn.AdaptiveMaxPool2d(1)

        # Classification Head
        # Fusion Input: Block 4 Features + Incidence Angle
        fusion_dim = channels[3] + 1
        hidden_dim = 512  # As per Lesson 40 "Current Best"

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=head_dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, inc_angle):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Image input [Batch, 3, 75, 75]
            inc_angle (torch.Tensor): Incidence angle input [Batch]
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Global Max Pooling
        x = self.global_pool(x).view(x.size(0), -1)

        # Prepare angle for concatenation
        angle = inc_angle.view(-1, 1)

        # Fusion
        fused = torch.cat([x, angle], dim=1)

        # Classification
        logits = self.head(fused)

        return logits
