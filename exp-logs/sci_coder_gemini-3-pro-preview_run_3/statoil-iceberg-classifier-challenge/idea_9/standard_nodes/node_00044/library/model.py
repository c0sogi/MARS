import torch
import torch.nn as nn
from library.config import Config


class SimpleCNN(nn.Module):
    """
    Simple 4-layer CNN architecture.
    Identified as robust for small, noisy radar datasets (Cite solution_lesson_node_00031).
    Features:
    - 4 Convolutional Blocks (64 -> 128 -> 128 -> 128) (Cite solution_lesson_node_00026)
    - Global Max Pooling (Cite solution_lesson_node_00005)
    - Late Fusion with Incidence Angle
    - Default Initialization (Cite solution_lesson_node_00041)
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Load hyperparameters from Config
        classifier_dropout = Config.CLASSIFIER_DROPOUT_RATE
        hidden_dim = Config.CLASSIFIER_HIDDEN_DIM
        input_channels = Config.IN_CHANNELS

        # Block 1: 3 -> 64
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: 64 -> 128
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3: 128 -> 128
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 4: 128 -> 128
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global Max Pooling
        self.global_pool = nn.AdaptiveMaxPool2d(1)

        # Classifier Head
        # 128 channels + 1 angle
        self.classifier = nn.Sequential(
            nn.Linear(128 + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=classifier_dropout),  # Cite solution_lesson_node_00017
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, angle):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Late Fusion
        angle = angle.view(-1, 1)
        x = torch.cat([x, angle], dim=1)

        x = self.classifier(x)
        return x
