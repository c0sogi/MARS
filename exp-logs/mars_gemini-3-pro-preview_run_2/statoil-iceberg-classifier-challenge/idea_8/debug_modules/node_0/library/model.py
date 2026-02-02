import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ChannelAttention(nn.Module):
    """
    Channel Attention Module for CBAM.
    Aggregates spatial information using Avg and Max pooling, then processes
    via a shared MLP to compute channel-wise importance.
    """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        # Ensure hidden dimension is at least 4 to avoid information bottleneck in small layers
        hidden_planes = max(in_planes // ratio, 4)

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
    Aggregates channel information using Avg and Max pooling, then uses
    convolution to compute spatial importance.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = kernel_size // 2

        # Input channels = 2 (1 for AvgPool, 1 for MaxPool)
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


class DualPathBlock(nn.Module):
    """
    Dual-Path Multi-Scale Block.
    Parallel processing of local (3x3) and context (3x3 dilated) features,
    followed by fusion and attention refinement.
    """

    def __init__(self, in_channels, out_channels):
        super(DualPathBlock, self).__init__()

        # Split output channels between the two paths
        assert out_channels % 2 == 0, "out_channels must be divisible by 2"
        mid_channels = out_channels // 2

        # Path 1: Local Stream (Standard 3x3 Conv)
        self.local_conv = nn.Conv2d(
            in_channels, mid_channels, kernel_size=3, padding=1, bias=False
        )

        # Path 2: Context Stream (Dilated 3x3 Conv, Dilation=2 -> Receptive Field ~5x5)
        self.context_conv = nn.Conv2d(
            in_channels, mid_channels, kernel_size=3, padding=2, dilation=2, bias=False
        )

        # Fusion Processing
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Attention Refinement
        self.cbam = CBAM(out_channels)

    def forward(self, x):
        # Parallel execution
        local_feat = self.local_conv(x)
        context_feat = self.context_conv(x)

        # Fusion (Concatenation)
        out = torch.cat([local_feat, context_feat], dim=1)

        # Activation
        out = self.bn(out)
        out = self.relu(out)

        # Attention
        out = self.cbam(out)
        return out


class MSAHN(nn.Module):
    """
    Multi-Scale Attention Hybrid Network (MSA-HN).
    Integrates a dual-path visual backbone with a metadata processing branch.
    """

    def __init__(self):
        super(MSAHN, self).__init__()

        # ==========================
        # 1. Visual Backbone
        # ==========================
        # Input: (Batch, 3, 75, 75)

        # Stage 1: 3 -> 32
        self.stage1 = nn.Sequential(
            DualPathBlock(Config.IMG_CHANNELS, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 75 -> 37
        )

        # Stage 2: 32 -> 64
        self.stage2 = nn.Sequential(
            DualPathBlock(32, 64), nn.MaxPool2d(kernel_size=2, stride=2)  # 37 -> 18
        )

        # Stage 3: 64 -> 128
        self.stage3 = nn.Sequential(
            DualPathBlock(64, 128), nn.MaxPool2d(kernel_size=2, stride=2)  # 18 -> 9
        )

        # Stage 4: 128 -> 256
        self.stage4 = nn.Sequential(
            DualPathBlock(128, 256), nn.MaxPool2d(kernel_size=2, stride=2)  # 9 -> 4
        )

        # Flattened Dimension: 256 channels * 4 * 4 spatial
        self.visual_dim = 256 * 4 * 4

        # ==========================
        # 2. Metadata Branch
        # ==========================
        # Input: (Batch, 1)
        self.meta_dim = 32
        self.meta_mlp = nn.Sequential(
            nn.Linear(1, self.meta_dim),
            nn.BatchNorm1d(self.meta_dim),
            nn.ReLU(inplace=True),
        )

        # ==========================
        # 3. Fusion Head
        # ==========================
        fusion_dim = self.visual_dim + self.meta_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(256, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """
        Kaiming initialization for Conv layers and Xavier for Linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_img, x_inc):
        """
        Args:
            x_img: Image tensor of shape (Batch, 3, 75, 75)
            x_inc: Incidence angle tensor of shape (Batch, 1)
        """
        # Visual Path
        v = self.stage1(x_img)
        v = self.stage2(v)
        v = self.stage3(v)
        v = self.stage4(v)

        # Flatten preserving spatial structure (Batch, 4096)
        v = v.view(v.size(0), -1)

        # Metadata Path
        m = self.meta_mlp(x_inc)

        # Fusion
        combined = torch.cat([v, m], dim=1)

        # Classification
        logits = self.classifier(combined)

        return logits
