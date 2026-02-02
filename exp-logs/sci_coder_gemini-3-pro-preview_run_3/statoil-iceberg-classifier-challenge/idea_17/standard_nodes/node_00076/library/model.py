import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SimpleCNN(nn.Module):
    """
    Simple 4-layer CNN with Global Max Pooling.
    Cite solution_lesson_node_00031: Simplicity and Inductive Bias.
    Cite solution_lesson_node_00005: Global Max Pooling.
    Cite solution_lesson_node_00026: Channel Width Constraints (capped at 128).
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Configuration
        # Using [64, 128, 128, 128] as per Lesson 00026 and 00050
        widths = Config.CHANNEL_WIDTHS
        dropout_rate = Config.DROPOUT_RATE
        num_classes = Config.NUM_CLASSES

        # Layer 1: 75x75 -> 37x37
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                Config.IN_CHANNELS, widths[0], kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Layer 2: 37x37 -> 18x18
        self.layer2 = nn.Sequential(
            nn.Conv2d(widths[0], widths[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(widths[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Layer 3: 18x18 -> 9x9
        self.layer3 = nn.Sequential(
            nn.Conv2d(widths[1], widths[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(widths[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Layer 4: 9x9 -> 4x4
        self.layer4 = nn.Sequential(
            nn.Conv2d(widths[2], widths[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(widths[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global Max Pooling
        self.global_pool = nn.AdaptiveMaxPool2d(1)

        # Classification Head
        # Cite solution_lesson_node_00040: Simplicity in Dense Classifier Heads
        # Input: 128 (CNN) + 1 (Angle) = 129
        input_dim = widths[3] + 1
        hidden_dim = 512  # Single hidden layer

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

        # Cite solution_lesson_node_00045: Use default initialization (Kaiming Uniform)
        # No manual initialization needed as PyTorch defaults are correct.

    def forward(self, x, angle):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Late Fusion with Raw Incidence Angle
        # Cite solution_lesson_node_00057: Scale Alignment (Raw scale ~30-45 matches features)
        angle = angle.view(-1, 1)
        x = torch.cat([x, angle], dim=1)

        x = self.head(x)
        return x.squeeze(1)
