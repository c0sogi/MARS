import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SHMP_CNN(nn.Module):
    """
    SHMP-CNN Architecture (Cite solution_lesson_node_00001, solution_lesson_node_00005).

    Architecture:
    - 4 Convolutional Blocks (Conv -> BN -> ReLU -> MaxPool)
    - Channels: 64 -> 128 -> 128 -> 128 (Cite solution_lesson_node_00050)
    - Global Max Pooling on final block only (Cite solution_lesson_node_00035)
    - Late Fusion with Incidence Angle (Cite solution_lesson_node_00039)
    - Simple Dense Head (Cite solution_lesson_node_00040)
    """

    def __init__(self):
        super(SHMP_CNN, self).__init__()

        # Block 1: 75x75 -> 37x37
        self.block1 = nn.Sequential(
            nn.Conv2d(
                Config.IN_CHANNELS, Config.BLOCK_CHANNELS[0], kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(Config.BLOCK_CHANNELS[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: 37x37 -> 18x18
        # Early expansion to 128 channels (Cite solution_lesson_node_00050)
        self.block2 = nn.Sequential(
            nn.Conv2d(
                Config.BLOCK_CHANNELS[0],
                Config.BLOCK_CHANNELS[1],
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(Config.BLOCK_CHANNELS[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3: 18x18 -> 9x9
        self.block3 = nn.Sequential(
            nn.Conv2d(
                Config.BLOCK_CHANNELS[1],
                Config.BLOCK_CHANNELS[2],
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(Config.BLOCK_CHANNELS[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 4: 9x9 -> 4x4
        self.block4 = nn.Sequential(
            nn.Conv2d(
                Config.BLOCK_CHANNELS[2],
                Config.BLOCK_CHANNELS[3],
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(Config.BLOCK_CHANNELS[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classification Head
        # Input: 128 (from Pool) + 1 (Angle)
        self.fusion_dim = Config.BLOCK_CHANNELS[3] + 1

        self.fc1 = nn.Linear(self.fusion_dim, Config.FC_HIDDEN_DIM)
        self.dropout = nn.Dropout(p=Config.DROPOUT_RATE)
        self.fc2 = nn.Linear(Config.FC_HIDDEN_DIM, 1)

    def forward(self, x, angle):
        # Backbone
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Global Max Pooling (Cite solution_lesson_node_00005)
        # (B, 128, 4, 4) -> (B, 128)
        x = F.adaptive_max_pool2d(x, (1, 1)).view(x.size(0), -1)

        # Late Fusion (Cite solution_lesson_node_00039)
        angle = angle.view(-1, 1)
        x = torch.cat([x, angle], dim=1)

        # Dense Head
        x = self.fc1(x)
        x = F.relu(x)
        # Dropout after activation (Cite solution_lesson_node_00017)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
