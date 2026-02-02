import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import NUM_CHANNELS


class MaxSEModule(nn.Module):
    """
    Max-Squeeze-and-Excitation Module.

    Replaces the standard Global Average Pooling of SE blocks with Global Max Pooling.
    This modification ensures that high-intensity peak signals (characteristic of icebergs
    in SAR imagery) are preserved and drive the channel attention weights, rather than
    being diluted by the background noise.
    """

    def __init__(self, channels, reduction=16):
        super(MaxSEModule, self).__init__()
        # Ensure the reduced dimension is at least 1
        mid_channels = max(channels // reduction, 1)

        self.fc1 = nn.Linear(channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, channels)

    def forward(self, x):
        # Squeeze: Global Max Pooling
        # Input x: (Batch, Channels, H, W) -> Output: (Batch, Channels)
        squeeze = F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)

        # Excitation: Learn channel weights
        excitation = F.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation))

        # Reshape for broadcasting: (Batch, Channels, 1, 1)
        excitation = excitation.view(x.size(0), x.size(1), 1, 1)

        # Scale input features
        return x * excitation


class ConvBlock(nn.Module):
    """
    Plain Convolutional Block with Max-SE Attention.

    Structure: Conv2d -> BatchNorm -> LeakyReLU -> MaxSE -> MaxPool
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        # We explicitly retain bias=True to preserve initialization dynamics
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=True
        )
        self.bn = nn.BatchNorm2d(out_channels)
        # LeakyReLU with negative_slope=0.1 to preserve semantic negative values (radar shadows)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.se = MaxSEModule(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        x = self.pool(x)
        return x


class MAPCNN(nn.Module):
    """
    Max-Attentive Plain CNN (MAP-CNN).

    A custom 4-stage convolutional network optimized for SAR iceberg detection.
    Features:
    - Plain CNN backbone (no residuals) to filter speckle noise.
    - Max-SE attention to focus on peak signals.
    - Selective Hierarchical Pooling (Stage 3 + Stage 4) to capture multi-scale features.
    - Raw incidence angle fusion.
    """

    def __init__(self):
        super(MAPCNN, self).__init__()

        # Stage 1: 3 -> 64
        # Input is 75x75
        self.stage1 = ConvBlock(NUM_CHANNELS, 64)

        # Stage 2: 64 -> 128 (Early Expansion)
        # Input is ~37x37
        self.stage2 = ConvBlock(64, 128)

        # Stage 3: 128 -> 128
        # Input is ~18x18
        self.stage3 = ConvBlock(128, 128)

        # Stage 4: 128 -> 128
        # Input is ~9x9 -> Output is ~4x4
        self.stage4 = ConvBlock(128, 128)

        # Classification Head
        # We fuse:
        # 1. Global Max Pooled Stage 3 (128 features)
        # 2. Global Max Pooled Stage 4 (128 features)
        # 3. Raw Incidence Angle (1 feature)
        self.head_input_dim = 128 + 128 + 1
        self.head_hidden_dim = 256

        self.head_fc = nn.Linear(self.head_input_dim, self.head_hidden_dim)
        self.head_act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.head_drop = nn.Dropout(p=0.5)
        self.head_out = nn.Linear(self.head_hidden_dim, 1)

        # Note: Weights are initialized using PyTorch default (Kaiming Uniform)

    def forward(self, x, angle):
        # x: (Batch, 3, 75, 75)
        # angle: (Batch,)

        # Forward pass through backbone
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        # Selective Hierarchical Pooling (Global Max Pooling)
        # We extract features from the last two stages to combine medium-level
        # texture details with high-level abstract shapes.
        pool3 = F.adaptive_max_pool2d(x3, 1).view(x3.size(0), -1)  # (B, 128)
        pool4 = F.adaptive_max_pool2d(x4, 1).view(x4.size(0), -1)  # (B, 128)

        # Prepare incidence angle
        angle = angle.view(-1, 1)  # (B, 1)

        # Feature Fusion
        # Concatenate pooled features and raw angle
        fused = torch.cat([pool3, pool4, angle], dim=1)  # (B, 257)

        # Classification Head
        h = self.head_fc(fused)
        h = self.head_act(h)
        h = self.head_drop(h)
        logits = self.head_out(h)

        # Return shape (Batch,) to match target shape in BCEWithLogitsLoss
        return logits.squeeze(1)
