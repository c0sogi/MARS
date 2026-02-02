import torch
import torch.nn as nn
from library.config import Config


class ShallowCNN(nn.Module):
    """
    Shallow VGG-Style Convolutional Neural Network for Audio Tagging.

    This model processes 2D Log-Mel Spectrograms. It consists of four convolutional
    blocks followed by global max pooling and a linear classification head.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, in_channels=1):
        """
        Initialize the ShallowCNN model.

        Args:
            num_classes (int): Number of output classes (default: 80).
            in_channels (int): Number of input channels (default: 1 for mono spectrograms).
        """
        super(ShallowCNN, self).__init__()

        # Block 1: 1 -> 64 channels
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: 64 -> 128 channels
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3: 128 -> 256 channels
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 4: 256 -> 512 channels
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global Max Pooling
        # Aggregates features over the Time and Frequency dimensions
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))

        # Classification Head
        # Maps the aggregated features (512) to the class probabilities (80)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input spectrogram of shape (Batch, Channels, Freq, Time).

        Returns:
            torch.Tensor: Output logits of shape (Batch, Num_Classes).
        """
        # Feature Extraction
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Aggregation
        x = self.global_pool(x)

        # Flatten: (Batch, 512, 1, 1) -> (Batch, 512)
        x = torch.flatten(x, 1)

        # Classification
        x = self.fc(x)

        return x
