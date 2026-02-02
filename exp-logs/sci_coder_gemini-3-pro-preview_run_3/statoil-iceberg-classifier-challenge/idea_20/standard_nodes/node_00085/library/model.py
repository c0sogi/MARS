import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SimpleCNN(nn.Module):
    """
    SimpleCNN Architecture.

    References:
    - Cite {lesson_node_00026}: Structural Regularization via Channel Width Constraints.
    - Cite {lesson_node_00050}: Early Channel Expansion (64 -> 128).
    - Cite {lesson_node_00007}: Global Max Pooling.
    - Cite {lesson_node_00039}: Late Fusion.
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Cite {lesson_node_00076}: Keep bias=True
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Cite {lesson_node_00040}: Single hidden layer
        # Cite {lesson_node_00017}: Dropout after FC
        self.classifier = nn.Sequential(
            nn.Linear(128 + 1, 512),  # 128 features + 1 angle
            nn.ReLU(inplace=True),
            nn.Dropout(Config.FC_DROPOUT),
            nn.Linear(512, 1),
        )

        # Cite {lesson_node_00078}: Default initialization (Kaiming Uniform with LeakyReLU slope)
        # is better than explicit Kaiming Normal for ReLU on this dataset.

    def forward(self, x, angle):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Cite {lesson_node_00007}: Global Max Pooling
        x = F.adaptive_max_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        # Cite {lesson_node_00039}: Late Fusion
        # Cite {lesson_node_00057}: Raw angle fusion
        angle = angle.view(-1, 1)
        x = torch.cat([x, angle], dim=1)

        x = self.classifier(x)
        return x.squeeze(1)
