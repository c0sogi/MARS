import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import NUM_FILTERS, DROPOUT_RATE, NUM_CHANNELS


class ChannelAttention(nn.Module):
    """
    Channel Attention Module for CBAM.
    Aggregates spatial information using both Average and Max pooling,
    then processes via a shared MLP to generate channel weights.
    """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Ensure hidden layer has reasonable size
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
    Aggregates channel information using Average and Max pooling along the channel axis,
    then processes via a convolution to generate spatial weights.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pool across channels
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


class DualPooling(nn.Module):
    """
    Dual-Stream Pooling Module.
    Applies Max Pooling (for peaks) and Min Pooling (for shadows/troughs)
    and concatenates the results along the channel dimension.
    """

    def __init__(self):
        super(DualPooling, self).__init__()
        self.max_pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        # Max Pooling
        x_max = self.max_pool(x)
        # Min Pooling: Simulated via -MaxPool(-x)
        x_min = -self.max_pool(-x)
        # Concatenate: (N, C, H, W) -> (N, 2C, H/2, W/2)
        return torch.cat([x_max, x_min], dim=1)


class WideBlock(nn.Module):
    """
    Wide-Body Delayed-Integration Block.
    Structure: Conv (Wide) -> BN -> ReLU -> CBAM -> DualPooling.
    """

    def __init__(self, in_channels, out_filters=NUM_FILTERS):
        super(WideBlock, self).__init__()

        # Wide Convolution: Maps input to fixed wide filter count (128)
        self.conv = nn.Conv2d(in_channels, out_filters, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_filters)
        self.relu = nn.ReLU(inplace=True)

        # CBAM Attention (applied strictly before pooling)
        self.cbam = CBAM(out_filters)

        # Dual Pooling (Expands channels 128 -> 256)
        self.pool = DualPooling()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        # Refine features
        x = self.cbam(x)

        # Downsample and expand channels
        x = self.pool(x)
        return x


class MetadataNet(nn.Module):
    """
    Simple MLP to process the scalar incidence angle metadata.
    """

    def __init__(self, input_dim=1, output_dim=32):
        super(MetadataNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Linear(16, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Ensure input is (Batch, 1)
        if x.dim() == 1:
            x = x.unsqueeze(1)
        return self.net(x)


class RDPWBN(nn.Module):
    """
    Robust Dual-Path Wide-Body Network (RDP-WBN).

    Architecture:
    1. Visual Branch: 4 Stages of WideBlocks with DualPooling.
    2. Readout:
       - Path A: Spatial Context Stream (Conv + Flatten) - Retains structure.
       - Path B: Robust Intensity Stream (Global Avg Pooling) - Invariant stats.
    3. Metadata Branch: MLP for incidence angle.
    4. Fusion Head: Concatenation + Dense Layers.
    """

    def __init__(self):
        super(RDPWBN, self).__init__()

        # --- Visual Branch ---
        # Block 1: Input 3 -> 128 -> Pool -> 256
        self.block1 = WideBlock(NUM_CHANNELS, NUM_FILTERS)

        # Block 2: Input 256 -> 128 -> Pool -> 256
        self.block2 = WideBlock(NUM_FILTERS * 2, NUM_FILTERS)

        # Block 3: Input 256 -> 128 -> Pool -> 256
        self.block3 = WideBlock(NUM_FILTERS * 2, NUM_FILTERS)

        # Block 4: Input 256 -> 128 -> Pool -> 256
        self.block4 = WideBlock(NUM_FILTERS * 2, NUM_FILTERS)

        # Output of Block 4 is (Batch, 256, 4, 4) given 75x75 input

        # --- Robust Dual-Path Readout ---
        # Path A: Spatial Context Stream
        # Reduces channels but preserves spatial dims (4x4)
        self.path_a_conv = nn.Conv2d(NUM_FILTERS * 2, 64, kernel_size=3, padding=1)
        # Output: 64 * 4 * 4 = 1024

        # Path B: Robust Intensity Stream
        # Global Average Pooling to capture invariant signal statistics
        self.path_b_pool = nn.AdaptiveAvgPool2d(1)
        # Output: 256 * 1 * 1 = 256

        # --- Metadata Branch ---
        self.meta_net = MetadataNet()

        # --- Fusion Head ---
        # Visual (1024 + 256) + Meta (32) = 1312
        fusion_dim = 1024 + 256 + 32

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(512, 1),  # Logits output
        )

    def forward(self, x_img, x_angle):
        # Visual Backbone
        x = self.block1(x_img)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)  # Shape: (B, 256, 4, 4)

        # Readout Path A: Spatial Context
        xa = self.path_a_conv(x)  # (B, 64, 4, 4)
        xa = F.relu(xa)
        xa = xa.view(xa.size(0), -1)  # Flatten -> (B, 1024)

        # Readout Path B: Robust Intensity
        xb = self.path_b_pool(x)  # (B, 256, 1, 1)
        xb = xb.view(xb.size(0), -1)  # Flatten -> (B, 256)

        # Fuse Visual Features
        x_visual = torch.cat([xa, xb], dim=1)  # (B, 1280)

        # Metadata Branch
        x_meta = self.meta_net(x_angle)  # (B, 32)

        # Global Fusion
        x_final = torch.cat([x_visual, x_meta], dim=1)  # (B, 1312)

        # Classification Head
        logits = self.head(x_final)

        return logits
