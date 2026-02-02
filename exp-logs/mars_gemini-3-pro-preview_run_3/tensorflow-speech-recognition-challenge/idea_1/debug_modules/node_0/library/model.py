import torch
import torch.nn as nn
from library import config


class SimpleAudioCNN(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        """
        A simple 2D Convolutional Neural Network for Audio Classification.

        Args:
            num_classes (int): Number of output classes. Defaults to config.NUM_CLASSES.
        """
        super(SimpleAudioCNN, self).__init__()

        # Helper function to create a standard Conv block
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            )

        # Input shape: (Batch, 1, n_mels, time_steps)
        # Approx: (Batch, 1, 40, 101)

        # Block 1
        # Output: (32, 20, 50)
        self.block1 = conv_block(1, 32)

        # Block 2
        # Output: (64, 10, 25)
        self.block2 = conv_block(32, 64)

        # Block 3
        # Output: (128, 5, 12)
        self.block3 = conv_block(64, 128)

        # Block 4
        # Output: (256, 2, 6)
        self.block4 = conv_block(128, 256)

        # Global Average Pooling
        # Collapses spatial dims (H, W) -> (1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 1, n_mels, time_steps).

        Returns:
            torch.Tensor: Logits of shape (Batch, num_classes).
        """
        # Feature extraction
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Global pooling
        x = self.global_pool(x)

        # Flatten: (Batch, 256, 1, 1) -> (Batch, 256)
        x = torch.flatten(x, 1)

        # Classification
        x = self.fc(x)

        return x
