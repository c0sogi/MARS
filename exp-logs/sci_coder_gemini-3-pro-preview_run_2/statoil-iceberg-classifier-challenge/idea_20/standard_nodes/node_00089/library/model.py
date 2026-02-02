import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ChannelAttention(nn.Module):
    """
    Channel Attention Module for CBAM.
    Aggregates spatial information using AvgPool and MaxPool, then processes
    via a shared MLP to generate channel weights.
    """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # Ensure hidden planes is at least 1
        hidden_planes = max(in_planes // ratio, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden_planes, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module for CBAM.
    Aggregates channel information using AvgPool and MaxPool along the channel axis,
    then processes via a convolution to generate spatial weights.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return x * self.sigmoid(out)


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module.
    Sequentially applies Channel Attention and Spatial Attention.
    """

    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class DualStreamPooling(nn.Module):
    """
    Dual-Stream Pooling.
    Concatenates Max Pooling (Peaks) and Min Pooling (Shadows).
    Doubles the channel dimension.
    """

    def __init__(self):
        super(DualStreamPooling, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        # Standard Max Pooling
        x_max = self.pool(x)

        # Min Pooling implemented via Max Pooling on negated input
        # min(x) = -max(-x)
        x_min = -self.pool(-x)

        return torch.cat([x_max, x_min], dim=1)


class DPCNet(nn.Module):
    """
    Dual-Pooling Contracted Network (DPCNet).
    Implements channel contraction in the final stage and preserves 4x4 spatial grid.
    Cite Lesson 41, Lesson 21.
    """

    def __init__(self):
        super(DPCNet, self).__init__()

        # --- Visual Branch ---
        # Input: 3 Channels (HH, HV, Avg)

        # Stage 1
        # In: 3 -> Conv(64) -> CBAM -> DualPool -> Out: 128
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.cbam1 = CBAMBlock(64)
        self.pool1 = DualStreamPooling()

        # Stage 2
        # In: 128 -> Conv(128) -> CBAM -> DualPool -> Out: 256
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.cbam2 = CBAMBlock(128)
        self.pool2 = DualStreamPooling()

        # Stage 3
        # In: 256 -> Conv(128) -> CBAM -> DualPool -> Out: 256
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.cbam3 = CBAMBlock(128)
        self.pool3 = DualStreamPooling()

        # Stage 4: Channel Contraction (Cite Lesson 41)
        # In: 256 -> Conv(64) -> CBAM -> DualPool -> Out: 128
        self.conv4 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.cbam4 = CBAMBlock(64)
        self.pool4 = DualStreamPooling()

        # --- Metadata Branch ---
        # Processes 'inc_angle'
        self.meta_fc1 = nn.Linear(1, 32)
        self.meta_bn1 = nn.BatchNorm1d(32)
        self.meta_fc2 = nn.Linear(32, 32)
        self.meta_bn2 = nn.BatchNorm1d(32)

        # --- Fusion Head ---
        # Visual: 128 channels * 4 * 4 spatial = 2048
        # Meta: 32
        # Total: 2080
        self.fusion_fc1 = nn.Linear(2048 + 32, 512)
        self.fusion_bn1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)
        self.fusion_fc2 = nn.Linear(512, 1)

    def forward(self, x, inc_angle):
        # --- Visual Branch ---

        # Stage 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.cbam1(x)  # Pre-Pooling Attention
        x = self.pool1(x)

        # Stage 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.cbam2(x)
        x = self.pool2(x)

        # Stage 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.cbam3(x)
        x = self.pool3(x)

        # Stage 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.cbam4(x)
        x = self.pool4(x)

        # Flatten directly (Cite Lesson 43, Lesson 21)
        # (B, 128, 4, 4) -> (B, 2048)
        x_vis = x.view(x.size(0), -1)

        # --- Metadata Branch ---
        # Ensure inc_angle has shape (B, 1)
        if inc_angle.dim() == 1:
            inc_angle = inc_angle.unsqueeze(1)

        x_meta = self.meta_fc1(inc_angle)
        x_meta = self.meta_bn1(x_meta)
        x_meta = F.relu(x_meta)

        x_meta = self.meta_fc2(x_meta)
        x_meta = self.meta_bn2(x_meta)
        x_meta = F.relu(x_meta)

        # --- Fusion ---
        x_fused = torch.cat([x_vis, x_meta], dim=1)

        x_fused = self.fusion_fc1(x_fused)
        x_fused = self.fusion_bn1(x_fused)
        x_fused = F.relu(x_fused)
        x_fused = self.dropout(x_fused)

        out = self.fusion_fc2(x_fused)

        return torch.sigmoid(out)
