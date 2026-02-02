import torch
import torch.nn as nn
import torch.nn.functional as F


class DualPooling(nn.Module):
    """
    Dual-Stream Pooling: Applies Max Pooling (Peaks) and Min Pooling (Shadows)
    and concatenates the outputs.
    """

    def __init__(self, kernel_size=2, stride=2):
        super(DualPooling, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        # Max pooling
        max_p = self.pool(x)
        # Min pooling implemented as -Max(-x)
        min_p = -self.pool(-x)
        # Concatenate along channel dimension
        return torch.cat([max_p, min_p], dim=1)


class ChannelAttention(nn.Module):
    """
    CBAM Channel Attention Module using Mixed Pooling (Max + Avg).
    """

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
    """
    CBAM Spatial Attention Module.
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
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    """

    def __init__(self, planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class WideBodyBlock(nn.Module):
    """
    Wide-Body Block with Delayed Integration.
    Structure: Conv(in->128) -> BN -> ReLU -> CBAM -> DualPooling.
    """

    def __init__(self, in_channels, out_channels=128):
        super(WideBodyBlock, self).__init__()

        # Wide Convolution: Maps input to fixed width of 128
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Pre-Pooling Attention
        self.cbam = CBAM(out_channels)

        # Dual-Stream Pooling (Expands channels 128 -> 256)
        self.pool = DualPooling(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.cbam(x)
        x = self.pool(x)
        return x


class DN_WBN(nn.Module):
    """
    Deeply-Normalized Wide-Body Network.
    """

    def __init__(self):
        super(DN_WBN, self).__init__()

        # ==========================
        # 1. Visual Branch
        # ==========================
        # Stage 1: 3 -> 128 -> Pool -> 256 (37x37)
        self.stage1 = WideBodyBlock(3, 128)
        # Stage 2: 256 -> 128 -> Pool -> 256 (18x18)
        self.stage2 = WideBodyBlock(256, 128)
        # Stage 3: 256 -> 128 -> Pool -> 256 (9x9)
        self.stage3 = WideBodyBlock(256, 128)
        # Stage 4: 256 -> 128 -> Pool -> 256 (4x4)
        self.stage4 = WideBodyBlock(256, 128)

        # Normalized Dual-Path Readout
        # Path A: Spatial Context
        # Input: 256 x 4 x 4
        self.path_a_conv = nn.Conv2d(
            256, 48, kernel_size=3, padding=1, bias=False
        )  # 48 x 4 x 4
        self.path_a_bn = nn.BatchNorm1d(48 * 4 * 4)  # 768 features

        # Path B: Robust Intensity
        self.path_b_gap = nn.AdaptiveAvgPool2d(1)
        self.path_b_bn = nn.BatchNorm1d(256)

        # ==========================
        # 2. Metadata Branch
        # ==========================
        # Deep Normalized Embedding
        # Structure: Linear -> ReLU -> Linear -> BN -> ReLU
        self.meta_fc1 = nn.Linear(1, 32)
        self.meta_relu1 = nn.ReLU(inplace=True)
        self.meta_fc2 = nn.Linear(32, 32)
        self.meta_bn = nn.BatchNorm1d(32)
        self.meta_relu2 = nn.ReLU(inplace=True)

        # ==========================
        # 3. Fusion Head
        # ==========================
        # Input sizes: Path A (768) + Path B (256) + Meta (32) = 1056
        self.fusion_fc = nn.Linear(768 + 256 + 32, 512)
        self.fusion_bn = nn.BatchNorm1d(512)
        self.fusion_relu = nn.ReLU(inplace=True)
        self.fusion_dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(512, 1)

    def forward(self, x_img, x_angle):
        # --- Visual Branch ---
        x = self.stage1(x_img)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)  # Shape: (B, 256, 4, 4)

        # Path A: Spatial
        feat_a = self.path_a_conv(x)  # (B, 48, 4, 4)
        feat_a = feat_a.view(feat_a.size(0), -1)  # Flatten -> (B, 768)
        feat_a = self.path_a_bn(feat_a)

        # Path B: Intensity
        feat_b = self.path_b_gap(x)  # (B, 256, 1, 1)
        feat_b = feat_b.view(feat_b.size(0), -1)  # Flatten -> (B, 256)
        feat_b = self.path_b_bn(feat_b)

        # --- Metadata Branch ---
        # Ensure angle is (B, 1)
        if x_angle.dim() == 1:
            x_angle = x_angle.view(-1, 1)

        meta = self.meta_fc1(x_angle)
        meta = self.meta_bn1(meta)
        meta = self.meta_relu1(meta)
        meta = self.meta_fc2(meta)
        meta = self.meta_bn2(meta)
        meta = self.meta_relu2(meta)

        # --- Fusion ---
        combined = torch.cat([feat_a, feat_b, meta], dim=1)

        f = self.fusion_fc(combined)
        f = self.fusion_bn(f)
        f = self.fusion_relu(f)
        f = self.fusion_dropout(f)

        out = self.classifier(f)

        # Return logits (BCEWithLogitsLoss expected) or Sigmoid?
        # Usually models return logits.
        return torch.sigmoid(out).view(-1)
