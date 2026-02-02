import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SimpleCNN(nn.Module):
    """
    Simple 4-layer CNN with Global Max Pooling.
    Replaces DenseNet to prioritize base learner quality (Cite Lesson 00015).
    Structure aligned with 'Current Best Solution' from lessons.
    """

    def __init__(self, drop_rate=Config.DROP_RATE, fc_dim=Config.FC_DIM):
        super(SimpleCNN, self).__init__()

        # Layer 1: 64 filters
        self.conv1 = nn.Sequential(
            nn.Conv2d(Config.IN_CHANNELS, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Layer 2: 128 filters
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Layer 3: 128 filters
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Layer 4: 128 filters
        # Maintained width to avoid bottleneck (Cite Lesson 00006)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classifier
        # Global Max Pooling output (128) + Angle (1)
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(128 + 1, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(fc_dim, 1),
        )

    def forward(self, x, angle):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Global Max Pooling (Cite Lesson 00005, 00007)
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(x.size(0), -1)

        # Feature Fusion
        angle = angle.view(-1, 1)
        x = torch.cat([x, angle], dim=1)

        x = self.classifier(x)
        return x
