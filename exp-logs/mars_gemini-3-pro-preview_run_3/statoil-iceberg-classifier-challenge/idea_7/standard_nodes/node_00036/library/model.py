import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import CNN_CHANNELS, DENSE_UNITS, DROPOUT_RATE, NUM_CHANNELS


class ConvBlock(nn.Module):
    """
    A standard convolutional block: Conv2d -> BatchNorm -> ReLU.
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SimpleCNN(nn.Module):
    """
    Simple Convolutional Neural Network (SimpleCNN).

    Reverts the Hierarchical Max-Pooling design to a standard sequential CNN.
    Cite solution_lesson_node_00035: Avoiding hierarchical filtering reduces overfitting to noise.
    Cite solution_lesson_node_00026: Caps channel width to enforce structural regularization.

    Features:
    - 4 Convolutional Blocks.
    - Global Max Pooling applied ONLY after the final block.
    - Concatenation of the final feature vector with the incidence angle.
    - Shallow, dense classification head.
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Unpack channel configurations from config
        # Default: [64, 128, 128, 128]
        c0 = NUM_CHANNELS  # 3 (HH, HV, Avg)
        c1, c2, c3, c4 = CNN_CHANNELS

        # Define Convolutional Blocks
        self.block1 = ConvBlock(c0, c1)
        self.block2 = ConvBlock(c1, c2)
        self.block3 = ConvBlock(c2, c3)
        self.block4 = ConvBlock(c3, c4)

        # Downsampling layer (Max Pooling 2x2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate dimension of the fused feature vector
        # Only the final block output + 1 for incidence angle
        fusion_dim = c4 + 1

        # Classification Head
        # Cite solution_lesson_node_00017: Dropout applied after the first dense layer activation
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, DENSE_UNITS),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DROPOUT_RATE),
            nn.Linear(DENSE_UNITS, 1),
        )

    def forward(self, x, angle):
        """
        Args:
            x (torch.Tensor): Input images of shape (Batch, 3, 75, 75)
            angle (torch.Tensor): Incidence angles of shape (Batch, 1)

        Returns:
            torch.Tensor: Probability of iceberg (Batch, 1)
        """

        # --- Block 1 ---
        x = self.block1(x)
        x = self.pool(x)

        # --- Block 2 ---
        x = self.block2(x)
        x = self.pool(x)

        # --- Block 3 ---
        x = self.block3(x)
        x = self.pool(x)

        # --- Block 4 ---
        x = self.block4(x)

        # Global Max Pooling (Cite solution_lesson_node_00007: Max Pooling > Avg Pooling)
        # (B, C, H, W) -> (B, C)
        x = F.adaptive_max_pool2d(x, (1, 1)).view(x.size(0), -1)

        # --- Feature Fusion ---
        # Concatenate final pooled vector and the incidence angle
        features = torch.cat([x, angle], dim=1)

        # --- Classification ---
        logits = self.head(features)

        # Return probability
        return torch.sigmoid(logits)
