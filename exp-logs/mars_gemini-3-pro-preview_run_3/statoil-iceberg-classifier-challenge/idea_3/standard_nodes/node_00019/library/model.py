import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SimpleCNN(nn.Module):
    """
    A simple 4-layer CNN architecture optimized for small, low-res radar images.
    Cite solution_lesson_node_00015: Prefer simple, shallow architectures for this dataset.
    Cite solution_lesson_node_00006: Maintain channel width (128) in final layer.
    """

    def __init__(self, drop_rate=Config.DROP_RATE, fc_dim=Config.FC_DIM):
        super(SimpleCNN, self).__init__()

        # Block 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(Config.IN_CHANNELS, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 4
        # Cite solution_lesson_node_00006: Do not reduce channels here (keep 128).
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.drop_rate = drop_rate

        # Classification Head
        # Global Max Pooling results in 128 features (from last conv layer)
        self.fc1 = nn.Linear(128 + 1, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 1)

    def forward(self, x, angle):
        # Feature Extraction
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Global Max Pooling
        # Cite solution_lesson_node_00007: Use Max Pooling to capture signal peaks.
        out = F.max_pool2d(out, kernel_size=out.size()[2:])
        out = out.view(out.size(0), -1)

        # Feature Fusion
        angle = angle.view(-1, 1)
        out = torch.cat([out, angle], dim=1)

        # Dense Layers
        out = F.relu(self.fc1(out))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.fc2(out)

        return out
