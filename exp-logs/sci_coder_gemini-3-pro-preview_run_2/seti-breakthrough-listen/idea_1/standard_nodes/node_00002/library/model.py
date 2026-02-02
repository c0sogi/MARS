import torch
import torch.nn as nn
from library.config import Config


class ShallowCNN(nn.Module):
    """
    A shallow 4-layer Convolutional Neural Network for Technosignature Detection.

    Architecture:
    - Input: (Batch, 6, 273, 256)
    - 4 Blocks of [Conv2d -> BatchNorm -> ReLU -> MaxPool2d]
    - Global Average Pooling
    - Fully Connected Output Layer (Logits)
    """

    def __init__(self):
        super(ShallowCNN, self).__init__()

        # Input channels from config (default 6)
        in_channels = Config.NUM_CHANNELS

        # Block 1: 6 -> 32
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # Block 2: 32 -> 64
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # Block 3: 64 -> 128
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # Block 4: 128 -> 256
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # Global Aggregation
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Output Head
        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 6, 273, 256)

        Returns:
            torch.Tensor: Raw logits of shape (Batch, 1)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Global Average Pooling: (Batch, 256, H, W) -> (Batch, 256, 1, 1)
        x = self.global_pool(x)

        # Flatten: (Batch, 256, 1, 1) -> (Batch, 256)
        x = torch.flatten(x, 1)

        # Classifier: (Batch, 256) -> (Batch, 1)
        x = self.classifier(x)

        return x
