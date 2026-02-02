import torch
import torch.nn as nn
from library.config import Config


class SpectroCNN(nn.Module):
    """
    A Simple 2D Convolutional Neural Network for Audio Classification.
    Processes Log-Mel Spectrograms as 2D images.

    Architecture:
    - 4 Convolutional Blocks (Conv -> BN -> ReLU -> MaxPool)
    - Global Average Pooling
    - Linear Classifier
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, input_channels=1):
        """
        Args:
            num_classes (int): Number of output classes. Defaults to Config.NUM_CLASSES (12).
            input_channels (int): Number of input channels (1 for mono spectrograms).
        """
        super(SpectroCNN, self).__init__()

        # Feature Extractor
        # Input Shape: [Batch, 1, 64, 100] (approx)
        self.features = nn.Sequential(
            # Block 1: 1 -> 32
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classifier Head
        # Global Average Pooling reduces (Batch, 256, H, W) -> (Batch, 256, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Layer
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input spectrograms of shape [Batch, 1, n_mels, time].

        Returns:
            torch.Tensor: Logits of shape [Batch, num_classes].
        """
        # Feature extraction
        x = self.features(x)

        # Global pooling
        x = self.global_pool(x)

        # Flatten: [Batch, 256, 1, 1] -> [Batch, 256]
        x = torch.flatten(x, 1)

        # Classification
        logits = self.fc(x)

        return logits
