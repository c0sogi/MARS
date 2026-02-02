import torch
import torch.nn as nn
from library.config import MODEL_PARAMS


class SimpleCNN(nn.Module):
    """
    A simple 4-layer CNN architecture optimized for small, noisy datasets.
    Key features:
    - Early channel expansion (64 -> 128) (Cite Lesson 00050)
    - Global Max Pooling (Cite Lesson 00005, 00007)
    - Late Fusion of raw incidence angle (Cite Lesson 00039, 00057)
    - Shallow Dense Head (Cite Lesson 00040)
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        in_channels = MODEL_PARAMS["input_channels"]
        dropout_rate = MODEL_PARAMS["dropout_rate"]

        # Layer 1: 75x75 -> 37x37
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Layer 2: 37x37 -> 18x18
        # Early expansion to 128 channels (Cite Lesson 00050)
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Layer 3: 18x18 -> 9x9
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Layer 4: 9x9 -> 4x4
        # Cap channels at 128 (Cite Lesson 00026)
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global Max Pooling (Cite Lesson 00005, 00007)
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))

        # Classifier Head
        # 128 (CNN features) + 1 (Incidence Angle)
        self.classifier = nn.Sequential(
            nn.Linear(128 + 1, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),  # Dropout after activation (Cite Lesson 00017)
            nn.Linear(512, 1),
        )

        # Initialization: Default PyTorch (Kaiming Uniform) (Cite Lesson 00045)

    def forward(self, x, angle):
        # Feature Extraction
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global Pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten -> (B, 128)

        # Late Fusion (Cite Lesson 00039)
        # Concatenate raw angle (Cite Lesson 00057)
        angle = angle.view(-1, 1)
        x = torch.cat([x, angle], dim=1)

        # Classification
        x = self.classifier(x)
        return x
