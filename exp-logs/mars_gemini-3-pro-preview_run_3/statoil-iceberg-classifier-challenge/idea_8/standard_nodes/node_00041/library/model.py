import torch
import torch.nn as nn
from library.config import Config


class SimpleCNN(nn.Module):
    """
    A simple 4-layer CNN with Global Max Pooling and Late Fusion.
    Optimized based on Lesson 00039 (Late Fusion) and Lesson 00026 (Channel Constraints).
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Layer 1: 64 channels
        self.layer1 = nn.Sequential(
            nn.Conv2d(Config.IN_CHANNELS, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Layer 2: 128 channels
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Layer 3: 128 channels
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Layer 4: 128 channels + Global Max Pooling (Cite Lesson 00007)
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1),
        )

        # Dense Head with Late Fusion
        # Input: 128 (image features) + 1 (angle)
        self.fc = nn.Sequential(
            nn.Linear(128 + 1, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(512, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, angle):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Flatten: (B, 128, 1, 1) -> (B, 128)
        x = x.view(x.size(0), -1)

        # Late Fusion (Cite Lesson 00039)
        angle = angle.view(-1, 1)
        x = torch.cat([x, angle], dim=1)

        x = self.fc(x)
        return x
