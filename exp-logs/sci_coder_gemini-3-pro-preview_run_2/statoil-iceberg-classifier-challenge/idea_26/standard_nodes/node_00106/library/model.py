import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class ChannelAttention(nn.Module):
    """
    Channel Attention Module for CBAM.
    Uses Mixed Pooling (Avg + Max) followed by a shared MLP.
    """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        # Reduction ratio for parameter efficiency
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
    Spatial Attention Module for CBAM.
    Uses channel-wise pooling (Avg + Max) followed by a 7x7 convolution.
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

    def __init__(self, planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class SpatiallyIntegratedBlock(nn.Module):
    """
    Core building block of IDSW-Net.
    Features:
    1. Wide Convolution (3x3)
    2. Pre-Pooling Attention (CBAM)
    3. Dual-Stream Pooling (Max + Min) to preserve dynamic range
    4. Post-Pooling Spatial Integration (3x3 Conv) to compress and integrate
    """

    def __init__(self, in_channels, out_channels):
        super(SpatiallyIntegratedBlock, self).__init__()

        # 1. Wide Convolution
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 2. Pre-Pooling Attention (CBAM)
        self.cbam = CBAM(out_channels)

        # 3. Dual-Stream Pooling
        # Standard Max Pooling
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Min Pooling is implemented functionally in forward() as -MaxPool(-x)

        # 4. Post-Pooling Spatial Integration
        # Input channels = out_channels (Max) + out_channels (Min) = 2 * out_channels
        # Compresses back to out_channels while integrating spatial context
        self.integration_conv = nn.Conv2d(
            out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Wide Conv -> BN -> ReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Attention Refinement
        x = self.cbam(x)

        # Dual-Stream Pooling
        # Max Pooling (Peaks)
        x_max = self.max_pool(x)
        # Min Pooling (Shadows): Inverse max pooling
        x_min = -self.max_pool(-x)

        # Concatenate features
        x_cat = torch.cat([x_max, x_min], dim=1)

        # Spatial Integration & Compression
        out = self.integration_conv(x_cat)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class IDSWNet(nn.Module):
    """
    Integrated Dual-Stream Wide Network.
    Combines a wide visual backbone with a metadata branch and a hybrid readout interface.
    """

    def __init__(self):
        super(IDSWNet, self).__init__()

        # --- Visual Branch ---
        # Input: 75x75x3
        # Structure: 4 SpatiallyIntegratedBlocks
        # Spatial Dim reduction: 75 -> 37 -> 18 -> 9 -> 4

        self.block1 = SpatiallyIntegratedBlock(config.IN_CHANNELS, 64)
        self.block2 = SpatiallyIntegratedBlock(64, config.BASE_FILTERS)  # 128
        self.block3 = SpatiallyIntegratedBlock(config.BASE_FILTERS, config.BASE_FILTERS)
        self.block4 = SpatiallyIntegratedBlock(config.BASE_FILTERS, config.BASE_FILTERS)

        # --- Hybrid Readout Interface ---
        # Path A: Spatial Context (Quadrants)
        self.pool_spatial = nn.AdaptiveMaxPool2d((2, 2))
        # Path B: Intensity Peak (Global)
        self.pool_global = nn.AdaptiveMaxPool2d((1, 1))

        # Calculate flattened dimension
        # Path A: 128 * 2 * 2 = 512
        # Path B: 128 * 1 * 1 = 128
        # Total: 640
        visual_dim = (config.BASE_FILTERS * 4) + config.BASE_FILTERS

        # --- Metadata Branch ---
        self.use_inc_angle = config.USE_INC_ANGLE
        if self.use_inc_angle:
            self.meta_mlp = nn.Sequential(
                nn.Linear(1, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
            )
            fusion_dim = visual_dim + 32
        else:
            fusion_dim = visual_dim

        # --- Fusion Head ---
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),  # High dropout (0.5)
            nn.Linear(256, 1),
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

    def forward(self, x_img, x_meta=None):
        # Visual Branch
        x = self.block1(x_img)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Hybrid Readout
        # Path A (Spatial 2x2)
        p_spatial = self.pool_spatial(x)  # [B, 128, 2, 2]
        p_spatial = p_spatial.view(p_spatial.size(0), -1)  # [B, 512]

        # Path B (Global 1x1)
        p_global = self.pool_global(x)  # [B, 128, 1, 1]
        p_global = p_global.view(p_global.size(0), -1)  # [B, 128]

        # Fusion
        visual_vec = torch.cat([p_spatial, p_global], dim=1)  # [B, 640]

        # Metadata Branch
        if self.use_inc_angle and x_meta is not None:
            # Ensure meta is [B, 1]
            if x_meta.dim() == 1:
                x_meta = x_meta.unsqueeze(1)
            meta_vec = self.meta_mlp(x_meta)
            final_vec = torch.cat([visual_vec, meta_vec], dim=1)
        else:
            final_vec = visual_vec

        # Classifier
        logits = self.classifier(final_vec)
        return logits
