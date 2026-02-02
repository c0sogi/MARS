import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ChannelAttention(nn.Module):
    """
    Channel Attention Module for CBAM.
    Refines features by exploiting inter-channel relationships.
    """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        # Use Conv2d with kernel_size=1 to act as MLP on channels
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Generate channel attention map
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module for CBAM.
    Refines features by exploiting inter-spatial relationships.
    Uses Max and Avg pooling along the channel axis.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        # Input channels = 2 (Max Pool + Avg Pool)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Mixed Pooling (Max + Avg) along channel axis
        # Note: Explicitly avoiding Min-pooling here as per instructions
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Sequentially applies Channel and Spatial attention.
    """

    def __init__(self, planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class DualPooling(nn.Module):
    """
    Dual-Stream Pooling.
    Concatenates the output of Max Pooling and Min Pooling.
    Effectively doubles the number of channels.
    """

    def __init__(self, kernel_size=2, stride=2):
        super(DualPooling, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        # Min pooling is implemented via negative max pooling

    def forward(self, x):
        # Standard Max Pooling
        out_max = self.max_pool(x)
        # Min Pooling: -max_pool(-x)
        out_min = -self.max_pool(-x)
        # Concatenate along channel dimension
        return torch.cat([out_max, out_min], dim=1)


class WBMGNet(nn.Module):
    """
    Wide-Body Multi-Granularity Network (WBMG-Net).

    Features:
    - Wide backbone (64->128->128->128 filters).
    - CBAM attention before pooling.
    - DualPooling (Max+Min) doubling channel depth at each stage.
    - Decoupled Readout Head (Intensity vs Spatial).
    - Metadata fusion.
    """

    def __init__(self):
        super(WBMGNet, self).__init__()

        # ==========================================
        # Visual Branch (Wide-Body Dual-Pooling Backbone)
        # ==========================================

        # Stage 1
        # Input: (B, 3, 75, 75)
        self.conv1 = nn.Conv2d(Config.IN_CHANNELS, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.cbam1 = CBAM(64)
        self.pool1 = DualPooling()
        # Output: 64 filters * 2 (DualPool) = 128 channels. Size: 75 -> 37

        # Stage 2
        # Input: 128 channels
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.cbam2 = CBAM(128)
        self.pool2 = DualPooling()
        # Output: 128 filters * 2 = 256 channels. Size: 37 -> 18

        # Stage 3
        # Input: 256 channels
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.cbam3 = CBAM(128)
        self.pool3 = DualPooling()
        # Output: 128 filters * 2 = 256 channels. Size: 18 -> 9

        # Stage 4
        # Input: 256 channels
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.cbam4 = CBAM(128)
        self.pool4 = DualPooling()
        # Output: 128 filters * 2 = 256 channels. Size: 9 -> 4
        # Final Volume: (B, 256, 4, 4)

        # ==========================================
        # Structural Innovation: Multi-Granularity Readout Head
        # ==========================================

        # Path A: Global Intensity
        # Global Max Pooling to capture strongest Peak/Shadow
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        # Output dim: 256

        # Path B: Spatial Layout
        # 1x1 Conv compression followed by Flatten
        self.spatial_reduce = nn.Conv2d(256, 48, kernel_size=1)
        # Flattened dim: 48 * 4 * 4 = 768
        self.flat_size_b = 48 * 4 * 4

        # ==========================================
        # Metadata Branch
        # ==========================================
        self.meta_fc1 = nn.Linear(1, 16)
        self.meta_fc2 = nn.Linear(16, 32)
        # Output dim: 32

        # ==========================================
        # Fusion Head
        # ==========================================
        # Total Features: 256 (Path A) + 768 (Path B) + 32 (Meta) = 1056
        self.fusion_fc = nn.Linear(256 + self.flat_size_b + 32, 512)
        self.fusion_bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)
        self.classifier = nn.Linear(512, 1)

    def forward(self, x, inc_angle):
        # --- Backbone Forward ---
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.cbam1(x)
        x = self.pool1(x)

        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.cbam2(x)
        x = self.pool2(x)

        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.cbam3(x)
        x = self.pool3(x)

        # Block 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.cbam4(x)
        x = self.pool4(x)

        # Feature Volume: (B, 256, 4, 4)

        # --- Readout Forward ---
        # Path A: Intensity (B, 256)
        path_a = self.global_max_pool(x)
        path_a = path_a.view(path_a.size(0), -1)

        # Path B: Spatial (B, 768)
        path_b = self.spatial_reduce(x)  # (B, 48, 4, 4)
        path_b = path_b.view(path_b.size(0), -1)

        # --- Metadata Forward ---
        # Ensure inc_angle is (B, 1)
        if inc_angle.dim() == 1:
            inc_angle = inc_angle.unsqueeze(1)

        meta = F.relu(self.meta_fc1(inc_angle))
        meta = F.relu(self.meta_fc2(meta))  # (B, 32)

        # --- Fusion ---
        fused = torch.cat((path_a, path_b, meta), dim=1)

        out = self.fusion_fc(fused)
        out = self.fusion_bn(out)
        out = F.relu(out)
        out = self.dropout(out)

        # Return logits (BCEWithLogitsLoss expected in training loop)
        logits = self.classifier(out)
        return logits
