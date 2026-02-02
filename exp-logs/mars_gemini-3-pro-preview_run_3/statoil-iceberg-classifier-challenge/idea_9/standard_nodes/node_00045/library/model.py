import torch
import torch.nn as nn
from library.config import Config


class SimpleCNN(nn.Module):
    """
    Simple 4-layer CNN with Global Max Pooling.
    Optimized for small-scale radar data classification.
    Cite Lesson 00015: Simplicity and Inductive Bias.
    Cite Lesson 00007: Global Max Pooling.
    Cite Lesson 00026: Structural Regularization via Channel Width Constraints.
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Block 1: 3 -> 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Block 2: 64 -> 128
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Block 3: 128 -> 128
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Block 4: 128 -> 128
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Global Max Pooling (Cite Lesson 00005, 00007)
        self.global_pool = nn.AdaptiveMaxPool2d(1)

        # Classifier Head
        # Input: 128 (CNN features) + 1 (Incidence Angle)
        # Cite Lesson 00017: Avoid Input Dropout on Compact Feature Vectors
        # Cite Lesson 00040: Simplicity in Dense Classifier Heads (Single Hidden Layer)
        self.classifier = nn.Sequential(
            nn.Linear(129, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, angle):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        # Late Fusion
        angle = angle.view(-1, 1)
        x = torch.cat([x, angle], dim=1)

        x = self.classifier(x)
        return x
