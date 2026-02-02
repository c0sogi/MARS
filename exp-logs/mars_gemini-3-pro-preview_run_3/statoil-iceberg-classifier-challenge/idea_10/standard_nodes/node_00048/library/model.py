import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SimpleCNN(nn.Module):
    """
    Simple 4-layer CNN with aggressive spatial reduction and Global Max Pooling.
    Cite solution_lesson_node_00046: Prefer Aggressive Spatial Reduction.
    Cite solution_lesson_node_00026: Structural Regularization via Channel Width Constraints.
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Layer 1: 75x75 -> 37x37
        self.conv1 = nn.Sequential(
            nn.Conv2d(Config.IN_CHANNELS, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Layer 2: 37x37 -> 18x18
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Layer 3: 18x18 -> 9x9
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Layer 4: 9x9 -> 4x4
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classifier
        # Global Max Pooling (128 channels) + 1 Angle
        self.classifier = nn.Sequential(
            nn.Linear(128 + 1, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(Config.DROPOUT_FC),
            nn.Linear(512, 1),
        )

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, angle):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Global Max Pooling
        x = F.adaptive_max_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        # Late Fusion
        if angle.dim() == 1:
            angle = angle.unsqueeze(1)
        x = torch.cat([x, angle], dim=1)

        x = self.classifier(x)
        return x
