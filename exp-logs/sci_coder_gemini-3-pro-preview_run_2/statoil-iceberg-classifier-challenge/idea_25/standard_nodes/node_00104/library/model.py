import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class SIBlock(nn.Module):
    """
    Spatially-Integrated Block (SIBlock)
    Structure:
    1. Wide Conv (3x3, 128) -> BN -> ReLU
    2. CBAM Attention (Pre-Pooling)
    3. Dual-Stream Pooling (Max + Min) -> 256 channels
    4. Spatial Integration Compression (3x3 Conv, 256->128) -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels=128):
        super(SIBlock, self).__init__()

        # 1. Wide Convolution
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 2. CBAM
        self.cbam = CBAM(out_channels)

        # 3. Pooling is functional in forward (Max + Min)

        # 4. Spatial Integration Compression
        # Input channels = out_channels * 2 (due to concatenation of Max and Min pool)
        self.integration_conv = nn.Conv2d(
            out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Wide Conv -> BN -> ReLU
        out = F.relu(self.bn1(self.conv1(x)))

        # Attention
        out = self.cbam(out)

        # Dual-Stream Pooling
        # Max Pooling
        p_max = F.max_pool2d(out, kernel_size=2, stride=2)
        # Min Pooling (implemented as -MaxPool(-x))
        p_min = -F.max_pool2d(-out, kernel_size=2, stride=2)

        # Concatenate
        pooled = torch.cat([p_max, p_min], dim=1)

        # Spatial Integration Compression
        out = F.relu(self.bn2(self.integration_conv(pooled)))

        return out


class SIWBN(nn.Module):
    """
    Spatially-Integrated Wide-Body Network (SI-WBN)
    Updated with Channel Contraction and Spatial Preservation.
    """

    def __init__(self):
        super(SIWBN, self).__init__()

        filters = Config.FILTERS

        # --- Visual Backbone ---
        # Input: 75x75
        # Cite Lesson 49: Prioritize backbone width in middle layers
        # Cite Lesson 41: Contract channels in final layer
        self.layer1 = SIBlock(Config.INPUT_CHANNELS, filters[0])  # -> 37x37 (64)
        self.layer2 = SIBlock(filters[0], filters[1])  # -> 18x18 (128)
        self.layer3 = SIBlock(filters[1], filters[2])  # -> 9x9 (128)
        self.layer4 = SIBlock(filters[2], filters[3])  # -> 4x4 (64)

        # Readout: Flatten 4x4 grid directly
        # Cite Lesson 21: Retain coarse spatial geometry (4x4) instead of aggressive pooling
        self.visual_dim = filters[3] * 4 * 4  # 64 * 16 = 1024

        # --- Metadata Branch ---
        self.meta_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.meta_dim = 32

        # --- Fusion Head ---
        fusion_dim = self.visual_dim + self.meta_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(256, 1),
        )

    def forward(self, x_img, x_angle):
        # Visual Branch
        x = self.layer1(x_img)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Flatten directly to preserve 4x4 spatial grid (Cite Lesson 21)
        x = x.view(x.size(0), -1)

        # Metadata Branch
        # Ensure angle is (Batch, 1)
        if x_angle.dim() == 1:
            x_angle = x_angle.unsqueeze(1)

        m = self.meta_mlp(x_angle)

        # Fusion
        combined = torch.cat([x, m], dim=1)

        # Classification
        logits = self.head(combined)

        return torch.sigmoid(logits)
