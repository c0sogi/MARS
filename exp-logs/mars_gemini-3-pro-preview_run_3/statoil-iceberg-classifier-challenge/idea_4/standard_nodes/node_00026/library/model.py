import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    NUM_CHANNELS,
    CONV_FILTERS,
    DENSE_UNITS,
    DROPOUT_RATE,
    NUM_CLASSES,
)


class IcebergCNN(nn.Module):
    """
    Custom CNN for Iceberg detection.
    Fuses image features with incidence angle metadata.
    """

    def __init__(self):
        super(IcebergCNN, self).__init__()

        # --- Backbone ---
        # Block 1
        self.conv1 = nn.Conv2d(NUM_CHANNELS, CONV_FILTERS[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(CONV_FILTERS[0])

        # Block 2
        self.conv2 = nn.Conv2d(
            CONV_FILTERS[0], CONV_FILTERS[1], kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(CONV_FILTERS[1])

        # Block 3
        self.conv3 = nn.Conv2d(
            CONV_FILTERS[1], CONV_FILTERS[2], kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(CONV_FILTERS[2])

        # Block 4
        self.conv4 = nn.Conv2d(
            CONV_FILTERS[2], CONV_FILTERS[3], kernel_size=3, padding=1
        )
        self.bn4 = nn.BatchNorm2d(CONV_FILTERS[3])

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
        x = self.pool(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)

        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
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
