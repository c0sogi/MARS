import torch
import torch.nn as nn
from library.config import NUM_CLASSES


class Simple3DCNN(nn.Module):
    """
    A lightweight 3D CNN to process the 6-frame cadence snippet.
    Treats the 6 input channels as a temporal sequence (Depth=6).
    Cite solution_lesson_node_00006: Explicitly preserves temporal dimension.
    """

    def __init__(self, num_classes=1):
        super().__init__()
        # Input shape: (B, 6, H, W) -> will be unsqueezed to (B, 1, 6, H, W)

        self.features = nn.Sequential(
            # Block 1: Preserve Time (6), Downsample Space
            # Output: (32, 6, H/2, W/2)
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            # Block 2: Downsample Time (6->3), Downsample Space
            # Output: (64, 3, H/4, W/4)
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            # Block 3: Preserve Time (3), Downsample Space
            # Output: (128, 3, H/8, W/8)
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            # Block 4: Collapse Time (3->1), Downsample Space
            # Kernel depth 3 consumes the remaining time dimension.
            # Output: (256, 1, H/16, W/16)
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )

        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (B, 6, H, W)
        # Add channel dimension: (B, 1, 6, H, W)
        x = x.unsqueeze(1)

        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def get_model():
    """
    Factory function to return the configured model.
    """
    return Simple3DCNN(num_classes=NUM_CLASSES)
