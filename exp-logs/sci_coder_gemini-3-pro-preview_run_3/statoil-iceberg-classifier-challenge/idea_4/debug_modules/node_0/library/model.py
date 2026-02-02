import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    NUM_CHANNELS,
    CONV_FILTERS,
    SE_REDUCTION_RATIO,
    DENSE_UNITS,
    DROPOUT_RATE,
    NUM_CLASSES,
)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels.
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: Global Average Pooling
        y = self.avg_pool(x).view(b, c)
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        y = self.fc(y).view(b, c, 1, 1)
        # Scale: Reweight original features
        return x * y.expand_as(x)


class IcebergSECNN(nn.Module):
    """
    Custom CNN with Squeeze-and-Excitation blocks for Iceberg detection.
    Fuses image features with incidence angle metadata.
    """

    def __init__(self):
        super(IcebergSECNN, self).__init__()

        # --- Backbone ---
        # Block 1
        self.conv1 = nn.Conv2d(NUM_CHANNELS, CONV_FILTERS[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(CONV_FILTERS[0])
        self.se1 = SEBlock(CONV_FILTERS[0], SE_REDUCTION_RATIO)

        # Block 2
        self.conv2 = nn.Conv2d(
            CONV_FILTERS[0], CONV_FILTERS[1], kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(CONV_FILTERS[1])
        self.se2 = SEBlock(CONV_FILTERS[1], SE_REDUCTION_RATIO)

        # Block 3
        self.conv3 = nn.Conv2d(
            CONV_FILTERS[1], CONV_FILTERS[2], kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(CONV_FILTERS[2])
        self.se3 = SEBlock(CONV_FILTERS[2], SE_REDUCTION_RATIO)

        # Block 4
        self.conv4 = nn.Conv2d(
            CONV_FILTERS[2], CONV_FILTERS[3], kernel_size=3, padding=1
        )
        self.bn4 = nn.BatchNorm2d(CONV_FILTERS[3])
        self.se4 = SEBlock(CONV_FILTERS[3], SE_REDUCTION_RATIO)

        # Pooling to reduce spatial dimensions within blocks
        self.pool = nn.MaxPool2d(2, 2)

        # --- Classification Head ---
        # Input features = Final Conv Filters (from Global Max Pool) + 1 (Incidence Angle)
        self.fc1 = nn.Linear(CONV_FILTERS[3] + 1, DENSE_UNITS)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.fc2 = nn.Linear(DENSE_UNITS, NUM_CLASSES)

    def forward(self, x, angle):
        """
        Args:
            x (torch.Tensor): Image tensor of shape (Batch, Channels, Height, Width)
            angle (torch.Tensor): Incidence angle tensor of shape (Batch,) or (Batch, 1)
        """
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.se1(x)
        x = self.pool(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.se2(x)
        x = self.pool(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.se3(x)
        x = self.pool(x)

        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.se4(x)
        x = self.pool(x)

        # Global Max Pooling
        # Reduces (Batch, 512, H, W) -> (Batch, 512, 1, 1) -> (Batch, 512)
        x = F.adaptive_max_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        # Feature Fusion
        # Ensure angle is (Batch, 1) to concatenate
        angle = angle.view(-1, 1)
        x = torch.cat([x, angle], dim=1)

        # Dense Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        # Output Probability
        x = torch.sigmoid(x)

        return x
