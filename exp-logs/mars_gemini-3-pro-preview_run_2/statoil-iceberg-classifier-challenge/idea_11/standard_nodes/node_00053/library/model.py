import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class ChannelAttention(nn.Module):
    """
    Channel Attention Module for CBAM.
    Aggregates spatial information using AvgPool and MaxPool, then processes
    through a shared MLP to generate channel-wise attention weights.
    """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        # Use a reduction ratio to save parameters, but ensure at least some width
        hidden_planes = max(in_planes // ratio, 8)

        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden_planes, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module for CBAM.
    Aggregates channel information using AvgPool and MaxPool along the channel axis,
    then processes through a convolution layer to generate spatial attention weights.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Sequentially applies Channel Attention and Spatial Attention.
    """

    def __init__(self, planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class A2SHN(nn.Module):
    """
    Attention-Augmented Shallow Hybrid Network (A2SHN).
    Optimized for small-data radar classification (Cite Lesson 00049).

    Features:
    1. Wide Backbone: Prioritizes width over depth/complexity.
    2. Channel Contraction: Reduces filters in final conv layer to control dense params (Cite Lesson 00041).
    3. CBAM: Attention mechanisms for feature refinement (Cite Lesson 00029).
    """

    def __init__(self):
        super(A2SHN, self).__init__()

        # Configuration
        filters = config.BACKBONE_FILTERS  # [64, 128, 128, 64]
        dropout_rate = config.DROPOUT_RATE
        in_channels = config.IMG_CHANNELS

        # --- Visual Branch ---

        # Stage 1 (Stem): 64 filters, MaxPool
        self.stage1_conv = nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1)
        self.stage1_bn = nn.BatchNorm2d(filters[0])
        self.stage1_pool = nn.MaxPool2d(2, 2)  # 75 -> 37

        # Stage 2: 128 filters, CBAM, MaxPool
        self.stage2_conv = nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1)
        self.stage2_bn = nn.BatchNorm2d(filters[1])
        self.stage2_cbam = CBAM(filters[1])
        self.stage2_pool = nn.MaxPool2d(2, 2)  # 37 -> 18

        # Stage 3: 128 filters, CBAM, MaxPool
        self.stage3_conv = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1)
        self.stage3_bn = nn.BatchNorm2d(filters[2])
        self.stage3_cbam = CBAM(filters[2])
        self.stage3_pool = nn.MaxPool2d(2, 2)  # 18 -> 9

        # Stage 4 (Contraction): 64 filters, CBAM, MaxPool (Cite Lesson 00041)
        self.stage4_conv = nn.Conv2d(filters[2], filters[3], kernel_size=3, padding=1)
        self.stage4_bn = nn.BatchNorm2d(filters[3])
        self.stage4_cbam = CBAM(filters[3])
        self.stage4_pool = nn.MaxPool2d(2, 2)  # 9 -> 4

        # Flatten size calculation: 4 * 4 * 64 = 1024
        self.flatten_dim = 4 * 4 * filters[3]

        # --- Metadata Branch ---
        # Simple MLP for the scalar incidence angle
        self.meta_fc1 = nn.Linear(1, 16)
        self.meta_bn1 = nn.BatchNorm1d(16)
        self.meta_act = nn.ReLU()

        # --- Fusion Head ---
        # Concatenate visual (1024) + meta (16) = 1040
        fusion_dim = self.flatten_dim + 16

        self.fusion_fc = nn.Linear(fusion_dim, 128)
        self.fusion_bn = nn.BatchNorm1d(128)
        self.fusion_drop = nn.Dropout(dropout_rate)
        self.output_fc = nn.Linear(128, 1)

    def forward(self, x_img, x_meta):
        # --- Visual Branch Forward ---
        # Stage 1
        x = self.stage1_conv(x_img)
        x = self.stage1_bn(x)
        x = F.relu(x)
        x = self.stage1_pool(x)

        # Stage 2
        x = self.stage2_conv(x)
        x = self.stage2_bn(x)
        x = F.relu(x)
        x = self.stage2_cbam(x)
        x = self.stage2_pool(x)

        # Stage 3
        x = self.stage3_conv(x)
        x = self.stage3_bn(x)
        x = F.relu(x)
        x = self.stage3_cbam(x)
        x = self.stage3_pool(x)

        # Stage 4
        x = self.stage4_conv(x)
        x = self.stage4_bn(x)
        x = F.relu(x)
        x = self.stage4_cbam(x)
        x = self.stage4_pool(x)

        # Flatten
        x_visual = x.view(x.size(0), -1)

        # --- Metadata Branch Forward ---
        # Ensure x_meta is (Batch, 1)
        if x_meta.dim() == 1:
            x_meta = x_meta.unsqueeze(1)

        x_m = self.meta_fc1(x_meta)
        x_m = self.meta_bn1(x_m)
        x_m = self.meta_act(x_m)

        # --- Fusion ---
        combined = torch.cat((x_visual, x_m), dim=1)

        out = self.fusion_fc(combined)
        out = self.fusion_bn(out)
        out = F.relu(out)
        out = self.fusion_drop(out)

        logits = self.output_fc(out)

        return logits
