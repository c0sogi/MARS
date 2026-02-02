import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SAHCN(nn.Module):
    """
    Spatially-Aware Hybrid Convolutional Network (SAHCN).

    Replaces the DenseNet architecture with a standard CNN using Max Pooling
    to preserve signal peaks in SAR imagery (Cite solution_lesson_node_00021).
    Uses Flattening instead of Global Average Pooling to retain spatial structure.
    """

    def __init__(self):
        super(SAHCN, self).__init__()

        # --- Visual Branch (CNN) ---
        # Input: (3, 75, 75)

        # Block 1: 75 -> 37
        self.conv1 = nn.Sequential(
            nn.Conv2d(Config.IN_CHANNELS, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: 37 -> 18
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3: 18 -> 9
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 4: 9 -> 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Flattened dimension: 64 * 4 * 4 = 1024
        self.visual_out_dim = 64 * 4 * 4

        # --- Metadata Branch ---
        self.meta_hidden_dim = Config.META_HIDDEN_DIM
        self.meta_branch = nn.Sequential(
            nn.Linear(1, self.meta_hidden_dim),
            nn.BatchNorm1d(self.meta_hidden_dim),
            nn.ReLU(),
        )

        # --- Fusion Head ---
        # Concatenated dimension
        fusion_input_dim = self.visual_out_dim + self.meta_hidden_dim

        # Classification Head
        # Cite solution_lesson_node_00014: Reduced dropout to 0.2
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(256, Config.NUM_CLASSES),
            nn.Sigmoid(),
        )

    def forward(self, x_img, x_angle):
        """
        Args:
            x_img: (Batch, 3, 75, 75)
            x_angle: (Batch,) or (Batch, 1)
        """
        # 1. Visual Branch
        x = self.conv1(x_img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Flatten (Cite solution_lesson_node_00021: preserve spatial grid)
        x = torch.flatten(x, 1)

        # 2. Metadata Branch
        if x_angle.dim() == 1:
            x_angle = x_angle.unsqueeze(1)

        m = self.meta_branch(x_angle)

        # 3. Fusion
        combined = torch.cat([x, m], dim=1)

        # 4. Classification
        out = self.fusion_head(combined)

        return out
