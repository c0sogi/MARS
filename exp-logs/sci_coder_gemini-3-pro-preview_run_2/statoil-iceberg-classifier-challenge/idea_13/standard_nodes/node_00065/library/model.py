import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    INITIAL_FILTERS,
    DEEP_FILTERS,
    DROPOUT_RATE,
    NUM_CHANNELS,
)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP implemented as 1x1 Convolutions
        # Reduces channels by 'ratio' then expands back
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

        # Compresses 2 channels (Avg+Max) into 1 spatial map
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Sequentially applies Channel Attention then Spatial Attention.
    """

    def __init__(self, planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class WBPA_Net(nn.Module):
    """
    Wide-Body Projected Attention Network.
    Features a sustained-width backbone, pre-pooling CBAM attention,
    and a projection bottleneck to decouple spatial width from classification density.
    """

    def __init__(self):
        super(WBPA_Net, self).__init__()

        # ==========================
        # 1. Visual Branch (Backbone)
        # ==========================

        # Stage 1: Expand to Initial Filters
        self.stage1_conv = nn.Conv2d(
            NUM_CHANNELS, INITIAL_FILTERS, kernel_size=3, padding=1
        )
        self.stage1_bn = nn.BatchNorm2d(INITIAL_FILTERS)
        self.stage1_cbam = CBAM(INITIAL_FILTERS)

        # Stage 2: Expand to Deep Filters (Sustained Width starts)
        self.stage2_conv = nn.Conv2d(
            INITIAL_FILTERS, DEEP_FILTERS, kernel_size=3, padding=1
        )
        self.stage2_bn = nn.BatchNorm2d(DEEP_FILTERS)
        self.stage2_cbam = CBAM(DEEP_FILTERS)

        # Stage 3: Maintain Deep Filters
        self.stage3_conv = nn.Conv2d(
            DEEP_FILTERS, DEEP_FILTERS, kernel_size=3, padding=1
        )
        self.stage3_bn = nn.BatchNorm2d(DEEP_FILTERS)
        self.stage3_cbam = CBAM(DEEP_FILTERS)

        # Stage 4: Maintain Deep Filters
        self.stage4_conv = nn.Conv2d(
            DEEP_FILTERS, DEEP_FILTERS, kernel_size=3, padding=1
        )
        self.stage4_bn = nn.BatchNorm2d(DEEP_FILTERS)
        self.stage4_cbam = CBAM(DEEP_FILTERS)

        # Pooling Layer (Shared across stages)
        self.pool = nn.MaxPool2d(2, 2)

        # Structural Innovation: 1x1 Projection Bottleneck
        # Projects 128 channels down to 64 to compress feature depth before flattening
        # Input map size here will be 4x4 (after 4 poolings of 75x75)
        self.projection = nn.Conv2d(DEEP_FILTERS, PROJECTION_DIM, kernel_size=1)

        # ==========================
        # 2. Metadata Branch
        # ==========================
        self.meta_fc1 = nn.Linear(1, 16)
        self.meta_bn1 = nn.BatchNorm1d(16)
        self.meta_fc2 = nn.Linear(16, 32)
        self.meta_bn2 = nn.BatchNorm1d(32)

        # ==========================
        # 3. Fusion Head
        # ==========================
        # Calculate Flattened Visual Dimension
        # 75 -> 37 -> 18 -> 9 -> 4 (Spatial dim after 4 pools)
        # Channels: PROJECTION_DIM (64)
        self.visual_flat_dim = 4 * 4 * PROJECTION_DIM
        self.meta_dim = 32

        fusion_input_dim = self.visual_flat_dim + self.meta_dim

        self.head_fc1 = nn.Linear(fusion_input_dim, 256)
        self.head_bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.head_out = nn.Linear(256, 1)

        # Initialize Weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_img, x_angle):
        # --- Visual Branch ---
        # Stage 1
        x = self.stage1_conv(x_img)
        x = self.stage1_bn(x)
        x = F.relu(x)
        x = self.stage1_cbam(x)  # Attention BEFORE Pooling
        x = self.pool(x)

        # Stage 2
        x = self.stage2_conv(x)
        x = self.stage2_bn(x)
        x = F.relu(x)
        x = self.stage2_cbam(x)
        x = self.pool(x)

        # Stage 3
        x = self.stage3_conv(x)
        x = self.stage3_bn(x)
        x = F.relu(x)
        x = self.stage3_cbam(x)
        x = self.pool(x)

        # Stage 4
        x = self.stage4_conv(x)
        x = self.stage4_bn(x)
        x = F.relu(x)
        x = self.stage4_cbam(x)
        x = self.pool(x)

        # Projection & Flatten
        x = self.projection(x)
        x_visual = x.view(x.size(0), -1)

        # --- Metadata Branch ---
        # Ensure angle is (Batch, 1)
        x_angle = x_angle.view(-1, 1)

        x_m = self.meta_fc1(x_angle)
        x_m = self.meta_bn1(x_m)
        x_m = F.relu(x_m)
        x_m = self.meta_fc2(x_m)
        x_m = self.meta_bn2(x_m)
        x_m = F.relu(x_m)

        # --- Fusion ---
        x_fused = torch.cat((x_visual, x_m), dim=1)

        # Classification Head
        x_out = self.head_fc1(x_fused)
        x_out = self.head_bn1(x_out)
        x_out = F.relu(x_out)
        x_out = self.dropout(x_out)

        # Output Logits (Sigmoid to be applied externally)
        logits = self.head_out(x_out)

        return logits
