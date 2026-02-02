import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple 4-layer CNN optimized for low-resolution radar images.
    Implements lessons:
    - Cite 00009: Custom architecture over pre-trained.
    - Cite 00050: Early channel expansion (64->128).
    - Cite 00046: Aggressive spatial reduction via pooling.
    - Cite 00091: LeakyReLU and Dropout 0.5.
    - Cite 00076: Redundant biases kept for initialization dynamics.
    - Cite 00078: Default PyTorch initialization used.
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Layer 1: 3 -> 64
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Layer 2: 64 -> 128
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Layer 3: 128 -> 128
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Layer 4: 128 -> 128
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Classifier Head
        # Global Max Pooling results in 128 features
        # + 1 Incidence Angle
        self.classifier = nn.Sequential(
            nn.Linear(128 + 1, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )

    def forward(self, x, angle):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global Max Pooling
        x = F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)

        # Late Fusion with Raw Incidence Angle (Cite 00039, 00057)
        if angle.dim() == 1:
            angle = angle.unsqueeze(1)

        x = torch.cat([x, angle], dim=1)
        x = self.classifier(x)

        return x
