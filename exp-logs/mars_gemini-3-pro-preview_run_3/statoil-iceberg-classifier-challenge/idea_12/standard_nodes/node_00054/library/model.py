import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SimpleCNN(nn.Module):
    """
    A simple 4-layer CNN with Global Max Pooling and Late Fusion.
    Cite {solution_lesson_node_00031}: Simplicity and Inductive Bias in Low-Resource Radar Classification.
    Cite {solution_lesson_node_00046}: Prefer Aggressive Spatial Reduction.
    Cite {solution_lesson_node_00050}: Early Channel Expansion.
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Block 1: 75x75 -> 37x37
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: 37x37 -> 18x18
        # Early expansion to 128 channels (Cite {solution_lesson_node_00050})
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3: 18x18 -> 9x9
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 4: 9x9 -> 4x4
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global Max Pooling: 4x4 -> 1x1
        self.global_pool = nn.AdaptiveMaxPool2d(1)

        # Classification Head
        # 128 features from CNN + 1 incidence angle
        self.classifier = nn.Sequential(
            nn.Linear(128 + 1, 512),
            nn.ReLU(inplace=True),
            # Dropout after activation (Cite {solution_lesson_node_00017})
            nn.Dropout(0.2),
            nn.Linear(512, 1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # Kaiming Uniform Initialization (Default for PyTorch, but explicit here)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, angle):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Late Fusion
        angle = angle.view(-1, 1)
        x = torch.cat([x, angle], dim=1)

        x = self.classifier(x)
        return x
