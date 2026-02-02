import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM) for CBAM.
    Aggregates channel information using Global Avg Pooling and Global Max Pooling,
    processed by a shared MLP.
    """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        # Use Conv2d with kernel_size=1 to act as MLP on channel dimension
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
    """
    Spatial Attention Module (SAM) for CBAM.
    Aggregates spatial information using Channel-wise Avg Pooling and Max Pooling,
    processed by a convolution layer.
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


class DualStreamPool(nn.Module):
    """
    Dual-Stream Pooling.
    Concatenates outputs from Max Pooling and Min Pooling to capture
    both peaks and shadows/voids in SAR data.
    """

    def __init__(self):
        super(DualStreamPool, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        # Max pooling
        max_p = self.pool(x)
        # Min pooling: implemented as -max_pool(-x)
        min_p = -self.pool(-x)
        return torch.cat([max_p, min_p], dim=1)


class WideBlock(nn.Module):
    """
    WideBlock component for WB-DSN.
    Structure: Conv -> BN -> ReLU -> CBAM -> DualStreamPool.
    """

    def __init__(self, in_channels, out_channels):
        super(WideBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_channels)
        self.pool = DualStreamPool()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.cbam(x)  # Pre-pooling attention
        x = self.pool(x)
        return x


class WB_DSN(nn.Module):
    """
    Wide-Body Dual-Stream Network (WB-DSN).
    Features:
    - Sustained width visual backbone (64->128->128->128).
    - Dual-stream pooling (Max+Min) doubling effective depth at each stage.
    - CBAM attention mechanisms.
    - Separate metadata branch for incidence angle.
    - Regularized fusion head with high dropout.
    """

    def __init__(self):
        super(WB_DSN, self).__init__()

        # --- Visual Branch ---
        # Stage 1: Input 3 -> 64 filters.
        # Output after dual pool: 64 * 2 = 128 channels.
        # Size: 75 -> 37
        self.stage1 = WideBlock(3, 64)

        # Stage 2: Input 128 -> 128 filters.
        # Output after dual pool: 128 * 2 = 256 channels.
        # Size: 37 -> 18
        self.stage2 = WideBlock(128, 128)

        # Stage 3: Input 256 -> 128 filters.
        # Output after dual pool: 128 * 2 = 256 channels.
        # Size: 18 -> 9
        self.stage3 = WideBlock(256, 128)

        # Stage 4: Input 256 -> 128 filters.
        # Output after dual pool: 128 * 2 = 256 channels.
        # Size: 9 -> 4
        self.stage4 = WideBlock(256, 128)

        # --- Metadata Branch ---
        # Processes scalar inc_angle
        self.meta_fc1 = nn.Linear(1, 16)
        self.meta_relu1 = nn.ReLU(inplace=True)
        self.meta_fc2 = nn.Linear(16, 32)
        self.meta_relu2 = nn.ReLU(inplace=True)

        # --- Fusion Head ---
        # Visual flat: 256 channels * 4 * 4 spatial = 4096
        # Meta flat: 32
        # Total input: 4128
        self.fusion_fc = nn.Linear(4096 + 32, 512)
        self.fusion_bn = nn.BatchNorm1d(512)
        self.fusion_relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)  # High dropout as per specification
        self.output = nn.Linear(512, 1)

    def forward(self, x_img, x_meta):
        # Visual Branch
        x = self.stage1(x_img)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Flatten visual features
        x = x.view(x.size(0), -1)

        # Metadata Branch
        m = self.meta_fc1(x_meta)
        m = self.meta_relu1(m)
        m = self.meta_fc2(m)
        m = self.meta_relu2(m)

        # Fusion
        combined = torch.cat([x, m], dim=1)

        f = self.fusion_fc(combined)
        f = self.fusion_bn(f)
        f = self.fusion_relu(f)
        f = self.dropout(f)

        out = self.output(f)
        return out
