import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SimpleCNN(nn.Module):
    """
    Simple 4-Layer CNN with Global Max Pooling.
    Based on Lessons 5, 26, 40.
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        widths = Config.CHANNEL_WIDTHS  # [64, 128, 128, 128]

        # Layer 1: 75 -> 37
        self.layer1 = nn.Sequential(
            nn.Conv2d(Config.IN_CHANNELS, widths[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Layer 2: 37 -> 18
        self.layer2 = nn.Sequential(
            nn.Conv2d(widths[0], widths[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(widths[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Layer 3: 18 -> 9
        self.layer3 = nn.Sequential(
            nn.Conv2d(widths[1], widths[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(widths[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Layer 4: 9 -> 4
        self.layer4 = nn.Sequential(
            nn.Conv2d(widths[2], widths[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(widths[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global Max Pooling (Cite Lesson 7, 34)
        self.global_pool = nn.AdaptiveMaxPool2d(1)

        # Classifier Head (Cite Lesson 40)
        # 128 features + 1 angle
        self.classifier = nn.Sequential(
            nn.Linear(widths[3] + 1, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(Config.DROPOUT_RATE),  # Cite Lesson 17
            nn.Linear(512, Config.NUM_CLASSES),
        )

    def forward(self, x, angle):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Late Fusion (Cite Lesson 39, 57)
        angle = angle.view(-1, 1)
        x = torch.cat([x, angle], dim=1)

        x = self.classifier(x)
        return x.squeeze(1)
