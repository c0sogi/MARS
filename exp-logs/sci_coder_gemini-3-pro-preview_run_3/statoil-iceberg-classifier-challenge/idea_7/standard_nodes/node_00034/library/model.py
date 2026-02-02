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


class HMP_CNN(nn.Module):
    """
    Hierarchical Max-Pooling Convolutional Neural Network (HMP-CNN).

    Features:
    - 4 Convolutional Blocks.
    - Global Max Pooling applied to the output of *every* block to capture multi-scale peak intensities.
    - Concatenation of multi-scale features with the incidence angle.
    - Shallow, dense classification head.
    """

    def __init__(self):
        super(HMP_CNN, self).__init__()

        # Unpack channel configurations from config
        # Default: [64, 64, 128, 128]
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
        # Sum of all block output channels + 1 for incidence angle
        fusion_dim = c1 + c2 + c3 + c4 + 1

        # Classification Head
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
        # Input: (B, 3, 75, 75) -> Output: (B, 64, 75, 75)
        x1 = self.block1(x)
        # Global Max Pool 1: (B, 64)
        v1 = F.adaptive_max_pool2d(x1, (1, 1)).view(x1.size(0), -1)
        # Downsample for next block: (B, 64, 37, 37)
        x_in2 = self.pool(x1)

        # --- Block 2 ---
        # Input: (B, 64, 37, 37) -> Output: (B, 64, 37, 37)
        x2 = self.block2(x_in2)
        # Global Max Pool 2: (B, 64)
        v2 = F.adaptive_max_pool2d(x2, (1, 1)).view(x2.size(0), -1)
        # Downsample for next block: (B, 64, 18, 18)
        x_in3 = self.pool(x2)

        # --- Block 3 ---
        # Input: (B, 64, 18, 18) -> Output: (B, 128, 18, 18)
        x3 = self.block3(x_in3)
        # Global Max Pool 3: (B, 128)
        v3 = F.adaptive_max_pool2d(x3, (1, 1)).view(x3.size(0), -1)
        # Downsample for next block: (B, 128, 9, 9)
        x_in4 = self.pool(x3)

        # --- Block 4 ---
        # Input: (B, 128, 9, 9) -> Output: (B, 128, 9, 9)
        x4 = self.block4(x_in4)
        # Global Max Pool 4: (B, 128)
        v4 = F.adaptive_max_pool2d(x4, (1, 1)).view(x4.size(0), -1)

        # --- Feature Fusion ---
        # Concatenate all pooled vectors and the incidence angle
        # v1: (B, 64), v2: (B, 64), v3: (B, 128), v4: (B, 128), angle: (B, 1)
        features = torch.cat([v1, v2, v3, v4, angle], dim=1)

        # --- Classification ---
        logits = self.head(features)

        # Return probability
        return torch.sigmoid(logits)
