import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import MODEL_PARAMS


class SimpleCNN(nn.Module):
    """
    Shallow 4-layer CNN with Global Max Pooling and Late Fusion.
    Optimized for small, noisy radar datasets.
    Cite solution_lesson_node_00031, solution_lesson_node_00046
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        in_channels = MODEL_PARAMS["input_channels"]
        dropout_rate = MODEL_PARAMS["dropout_rate"]

        self.conv = nn.Sequential(
            # Block 1: 3 -> 64
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2: 64 -> 128
            # Early channel expansion to preserve texture - Cite solution_lesson_node_00050
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3: 128 -> 128
            # Cap width at 128 to prevent overfitting - Cite solution_lesson_node_00026
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 4: 128 -> 128
            # Aggressive spatial reduction before pooling - Cite solution_lesson_node_00046
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Global Max Pooling - Cite solution_lesson_node_00005, solution_lesson_node_00007
        self.pool = nn.AdaptiveMaxPool2d(1)

        # Classifier
        # Input features: 128 (CNN) + 1 (Angle) = 129
        # Single hidden layer is sufficient - Cite solution_lesson_node_00040
        self.classifier = nn.Sequential(
            nn.Linear(129, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(
                dropout_rate
            ),  # Dropout after activation - Cite solution_lesson_node_00017
            nn.Linear(512, 1),
        )

    def forward(self, x, angle):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten (B, 128)

        # Late Fusion of raw incidence angle
        # Match numerical magnitude of auxiliary inputs - Cite solution_lesson_node_00057
        # Prefer Late Fusion over Feature-wise Modulation - Cite solution_lesson_node_00039
        angle = angle.view(-1, 1)
        x = torch.cat([x, angle], dim=1)

        x = self.classifier(x)
        return x
