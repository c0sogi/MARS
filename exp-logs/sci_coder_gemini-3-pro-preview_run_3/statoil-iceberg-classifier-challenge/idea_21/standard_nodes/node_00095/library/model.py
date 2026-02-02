import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    A robust 4-layer CNN architecture optimized for small SAR datasets.
    Key features:
    - Early channel expansion (64 -> 128) (Cite solution_lesson_node_00050)
    - LeakyReLU activations (Cite solution_lesson_node_00091)
    - Single hidden layer classification head (Cite solution_lesson_node_00040)
    - Default PyTorch initialization (Cite solution_lesson_node_00078)
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Block 1: 3 -> 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Block 2: 64 -> 128 (Early expansion)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Block 3: 128 -> 128
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Block 4: 128 -> 128
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Classifier Head
        # Global Max Pooling -> 128 features
        # + 1 incidence angle = 129 features
        self.head = nn.Sequential(
            nn.Linear(129, 512),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 1),
        )

    def forward(self, x, angle):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Global Max Pooling (Cite solution_lesson_node_00046)
        x = F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)

        if angle.dim() == 1:
            angle = angle.unsqueeze(1)

        # Late Fusion
        x = torch.cat([x, angle], dim=1)
        return self.head(x)
