import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SHMP_CNN(nn.Module):
    """
    Selective Hierarchical Max-Pooling CNN (SHMP-CNN).

    Architecture:
    - 4 Convolutional Blocks (Conv -> BN -> ReLU -> MaxPool)
    - Channel sizes: 64 -> 64 -> 128 -> 128
    - Selective Hierarchical Pooling: Global Max Pooling on Block 3 and Block 4 outputs.
    - Feature Fusion: Concatenation of Pool(Block3) + Pool(Block4) + Incidence Angle.
    - Classification Head: Linear -> ReLU -> Dropout -> Linear -> Logits.
    """

    def __init__(self):
        super(SHMP_CNN, self).__init__()

        # ----------------------------------------------------------------------
        # Convolutional Backbone
        # ----------------------------------------------------------------------
        # Input: (Batch, 3, 75, 75)

        # Block 1
        # 75x75 -> 37x37
        self.block1 = nn.Sequential(
            nn.Conv2d(
                Config.IN_CHANNELS, Config.BLOCK_CHANNELS[0], kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(Config.BLOCK_CHANNELS[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2
        # 37x37 -> 18x18
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

        # Block 3
        # 18x18 -> 9x9
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

        # Block 4
        # 9x9 -> 4x4
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
        # Calculate input dimension for the dense layer
        # Block 3 output channels: 128
        # Block 4 output channels: 128
        # Incidence Angle: 1
        # Total: 128 + 128 + 1 = 257

        self.fusion_dim = Config.BLOCK_CHANNELS[2] + Config.BLOCK_CHANNELS[3] + 1

        self.fc1 = nn.Linear(self.fusion_dim, Config.FC_HIDDEN_DIM)
        self.dropout = nn.Dropout(p=Config.DROPOUT_RATE)
        self.fc2 = nn.Linear(Config.FC_HIDDEN_DIM, 1)  # Binary classification (logits)

    def forward(self, x, angle):
        """
        Args:
            x (torch.Tensor): Image input of shape (B, 3, 75, 75)
            angle (torch.Tensor): Incidence angle input of shape (B,)
        """

        # --- Backbone Forward Pass ---
        x1 = self.block1(x)
        x2 = self.block2(x1)

        # We need the output of Block 3 for hierarchical pooling
        x3 = self.block3(x2)

        # We need the output of Block 4 for hierarchical pooling
        x4 = self.block4(x3)

        # --- Selective Hierarchical Pooling ---
        # Global Max Pooling on Block 3 (B, 128, 9, 9) -> (B, 128)
        # Using adaptive_max_pool2d(1) is equivalent to global max pooling
        pool3 = F.adaptive_max_pool2d(x3, (1, 1)).view(x3.size(0), -1)

        # Global Max Pooling on Block 4 (B, 128, 4, 4) -> (B, 128)
        pool4 = F.adaptive_max_pool2d(x4, (1, 1)).view(x4.size(0), -1)

        # --- Feature Fusion ---
        # Reshape angle to (B, 1) to match dimensions
        angle = angle.view(-1, 1)

        # Concatenate: [Pool3, Pool4, Angle]
        fused = torch.cat([pool3, pool4, angle], dim=1)

        # --- Classification Head ---
        out = self.fc1(fused)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        # Return logits (B, 1)
        # Squeeze to (B,) if necessary, but usually Loss expects (B, 1) or (B) depending on implementation
        # We return (B, 1) to be safe and consistent with BCEWithLogitsLoss
        return out
