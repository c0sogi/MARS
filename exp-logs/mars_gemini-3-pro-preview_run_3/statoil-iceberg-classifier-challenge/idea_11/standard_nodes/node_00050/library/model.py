import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SimpleCNN(nn.Module):
    """
    Simple 4-Layer CNN with Global Max Pooling and Late Fusion.
    Cite solution_lesson_node_00035: Prefer Hierarchical Filtering (Standard CNN) over Multi-Scale Aggregation.

    Architecture:
    - 4 Convolutional Blocks (Conv -> BN -> ReLU -> MaxPool)
    - Channel sizes: 64 -> 64 -> 128 -> 128
    - Global Max Pooling on the final block only.
    - Feature Fusion: Concatenation of Pool(Block4) + Incidence Angle.
    - Classification Head: Linear -> ReLU -> Dropout -> Linear -> Logits.
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # ----------------------------------------------------------------------
        # Convolutional Backbone
        # ----------------------------------------------------------------------
        # Input: (Batch, 3, 75, 75)

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

        # ----------------------------------------------------------------------
        # Classification Head
        # ----------------------------------------------------------------------
        # Block 4 output channels: 128
        # Incidence Angle: 1
        # Total: 128 + 1 = 129
        self.fusion_dim = Config.BLOCK_CHANNELS[3] + 1

        # Cite solution_lesson_node_00040: Single hidden layer is optimal for small data
        self.fc1 = nn.Linear(self.fusion_dim, Config.FC_HIDDEN_DIM)
        # Cite solution_lesson_node_00017: Apply Dropout after activation, not on input
        self.dropout = nn.Dropout(p=Config.DROPOUT_RATE)
        self.fc2 = nn.Linear(Config.FC_HIDDEN_DIM, 1)

    def forward(self, x, angle):
        # Backbone
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Global Max Pooling
        # Cite solution_lesson_node_00007: Prefer Max Pooling for sparse targets
        x = F.adaptive_max_pool2d(x, (1, 1)).view(x.size(0), -1)

        # Late Fusion
        # Cite solution_lesson_node_00039: Late Fusion over Feature-wise Modulation
        angle = angle.view(-1, 1)
        fused = torch.cat([x, angle], dim=1)

        # Head
        out = self.fc1(fused)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out
