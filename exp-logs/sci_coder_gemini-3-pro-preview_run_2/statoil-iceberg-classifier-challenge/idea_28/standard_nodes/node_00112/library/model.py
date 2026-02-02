import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (part of CBAM).
    Aggregates global spatial information using both Avg and Max pooling.
    """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        # Ensure hidden planes is at least some minimal value (e.g., 4)
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
    Spatial Attention Module (part of CBAM).
    Aggregates channel information using both Avg and Max pooling.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        # Input channels = 2 (1 for AvgPool, 1 for MaxPool)
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
    Applies Max Pooling (Peaks) and Min Pooling (Shadows) and concatenates outputs.
    Doubles the channel dimension.
    """

    def __init__(self, kernel_size=2, stride=2):
        super(DualPooling, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size, stride=stride)
        # Min pooling is implemented via MaxPool on negated input

    def forward(self, x):
        x_max = self.max_pool(x)
        x_min = -self.max_pool(-x)
        return torch.cat([x_max, x_min], dim=1)


class DIDPBlock(nn.Module):
    """
    Delayed-Integration Dual-Pyramid Block.
    Structure: Conv(Wide) -> BN -> ReLU -> CBAM -> DualPooling.
    """

    def __init__(self, in_channels, out_channels):
        super(DIDPBlock, self).__init__()

        # Wide Convolution: 3x3 Convolution
        # This performs integration/compression at the start of the block
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Pre-Pooling Attention (CBAM)
        self.cbam = CBAM(out_channels)

        # Dual-Stream Pooling (Expands channels 2x)
        self.pool = DualPooling()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        # Refine features at high resolution before pooling
        x = self.cbam(x)

        # Pool and expand channels
        x = self.pool(x)
        return x


class DIDPNet(nn.Module):
    """
    Delayed-Integration Dual-Pyramid Network (DIDP-Net).

    Features:
    - Split-Branch Topology (Visual + Metadata).
    - Sustained Width Backbone (128 filters).
    - Delayed Integration (Compression at start of new receptive field).
    - Quadrant-Based Readout.
    - High Dropout Regularization.
    """

    def __init__(
        self, backbone_filters=Config.BACKBONE_FILTERS, dropout_rate=Config.DROPOUT_RATE
    ):
        super(DIDPNet, self).__init__()

        self.backbone_filters = backbone_filters

        # ==========================
        # Visual Branch
        # ==========================

        # Stage 1: Input (3) -> 128 filters.
        # Output after DualPooling: 128 * 2 = 256 channels.
        # Size: 75 -> 37
        self.stage1 = DIDPBlock(3, self.backbone_filters)

        # Stage 2: Input (256) -> 128 filters.
        # Output after DualPooling: 256 channels.
        # Size: 37 -> 18
        self.stage2 = DIDPBlock(self.backbone_filters * 2, self.backbone_filters)

        # Stage 3: Input (256) -> 128 filters.
        # Output after DualPooling: 256 channels.
        # Size: 18 -> 9
        self.stage3 = DIDPBlock(self.backbone_filters * 2, self.backbone_filters)

        # Stage 4: Input (256) -> 128 filters.
        # Output after DualPooling: 256 channels.
        # Size: 9 -> 4
        self.stage4 = DIDPBlock(self.backbone_filters * 2, self.backbone_filters)

        # Quadrant-Based Readout
        # Adaptive Max Pooling to 2x2 grid
        self.quadrant_pool = nn.AdaptiveMaxPool2d((2, 2))

        # Flattened Visual Vector: 256 channels * 2 * 2 = 1024
        self.visual_dim = (self.backbone_filters * 2) * 2 * 2

        # ==========================
        # Metadata Branch
        # ==========================
        # Process incidence angle
        self.meta_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.meta_dim = 32

        # ==========================
        # Fusion Head
        # ==========================
        fusion_dim = self.visual_dim + self.meta_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, angle):
        """
        Args:
            x (torch.Tensor): Image input (N, 3, 75, 75)
            angle (torch.Tensor): Incidence angle input (N,) or (N, 1)
        """
        # --- Visual Forward ---
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Quadrant Readout (N, 256, 4, 4) -> (N, 256, 2, 2)
        x = self.quadrant_pool(x)
        x = x.view(x.size(0), -1)  # Flatten -> (N, 1024)

        # --- Metadata Forward ---
        if angle.dim() == 1:
            angle = angle.view(-1, 1)

        m = self.meta_mlp(angle)

        # --- Fusion ---
        combined = torch.cat([x, m], dim=1)

        # --- Classification ---
        # Returns logits
        out = self.classifier(combined)

        return out
