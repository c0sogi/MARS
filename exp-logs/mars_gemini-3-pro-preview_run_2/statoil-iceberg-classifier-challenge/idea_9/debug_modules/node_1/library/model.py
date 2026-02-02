import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ChannelAttention(nn.Module):
    """
    Channel Attention Module for CBAM.
    Aggregates spatial information using both Average and Max pooling,
    then processes them through a shared MLP to generate channel weights.
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
    Spatial Attention Module for CBAM.
    Aggregates channel information using Average and Max pooling along the channel axis,
    then processes them through a convolution to generate a spatial attention map.
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
    Sequentially applies Channel Attention and then Spatial Attention.
    """

    def __init__(self, planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class CSPHN(nn.Module):
    """
    Contracted Spatial Pyramid Hybrid Network (CSP-HN).

    Architecture:
    1. Visual Branch: 4-layer CNN with CBAM and MaxPooling.
    2. Bottleneck: 1x1 Conv to reduce channel depth before flattening.
    3. SPP: Spatial Pyramid Pooling at 4x4, 2x2, and 1x1 scales.
    4. Metadata Branch: MLP for incidence angle.
    5. Fusion: Concatenation -> Dense -> Output.
    """

    def __init__(self):
        super(CSPHN, self).__init__()

        # ==========================
        # 1. Visual Backbone
        # ==========================
        # Input: 3 channels (HH, HV, Avg) -> 75x75

        # Block 1: 75x75 -> 37x37
        self.conv1 = nn.Conv2d(Config.IMG_CHANNELS, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.cbam1 = CBAM(32)

        # Block 2: 37x37 -> 18x18
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.cbam2 = CBAM(64)

        # Block 3: 18x18 -> 9x9
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.cbam3 = CBAM(128)

        # Block 4: 9x9 -> 4x4
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.cbam4 = CBAM(128)

        self.pool = nn.MaxPool2d(2, 2)

        # ==========================
        # 2. Channel Contraction Bottleneck
        # ==========================
        # Reduces 128 channels to Config.BOTTLENECK_CHANNELS (e.g., 32)
        # Spatial dim remains 4x4
        self.bottleneck = nn.Conv2d(128, Config.BOTTLENECK_CHANNELS, kernel_size=1)
        self.bn_bottle = nn.BatchNorm2d(Config.BOTTLENECK_CHANNELS)

        # ==========================
        # 3. Spatial Pyramid Pooling (SPP) Calculation
        # ==========================
        # Input to SPP is (Batch, Bottleneck_Channels, 4, 4)
        # Level 1: 4x4 (Identity) -> 16 spatial locations
        # Level 2: 2x2 Pooling -> 4 spatial locations
        # Level 3: 1x1 Pooling (Global) -> 1 spatial location
        # Total spatial elements = 16 + 4 + 1 = 21
        self.spp_dim = Config.BOTTLENECK_CHANNELS * 21

        # ==========================
        # 4. Metadata Branch
        # ==========================
        self.meta_fc1 = nn.Linear(1, 16)
        self.meta_bn1 = nn.BatchNorm1d(16)
        self.meta_act = nn.ReLU()

        # ==========================
        # 5. Fusion Head
        # ==========================
        fusion_dim = self.spp_dim + 16

        self.fc1 = nn.Linear(fusion_dim, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)

        self.fc2 = nn.Linear(256, 1)  # Binary classification logits

    def forward(self, x_img, x_meta):
        # --- Visual Branch ---
        # Block 1
        x = self.conv1(x_img)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.cbam1(x)
        x = self.pool(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.cbam2(x)
        x = self.pool(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.cbam3(x)
        x = self.pool(x)

        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.cbam4(x)
        x = self.pool(x)

        # Bottleneck (Channel Contraction)
        x = self.bottleneck(x)
        x = self.bn_bottle(x)
        x = F.relu(x)
        # Current Shape: (Batch, 32, 4, 4)

        # Spatial Pyramid Pooling
        # Level 1: 4x4 (Identity / MaxPool k=1)
        spp1 = x.view(x.size(0), -1)  # Flatten (Batch, 32*16)

        # Level 2: 2x2 Pooling
        # Input 4x4 -> Pool k=2, s=2 -> Output 2x2
        x_2x2 = F.max_pool2d(x, kernel_size=2, stride=2)
        spp2 = x_2x2.view(x.size(0), -1)  # Flatten (Batch, 32*4)

        # Level 3: 1x1 Pooling (Global Max)
        # Input 4x4 -> Pool k=4 -> Output 1x1
        x_1x1 = F.max_pool2d(x, kernel_size=4)
        spp3 = x_1x1.view(x.size(0), -1)  # Flatten (Batch, 32*1)

        # Concatenate SPP features
        visual_feat = torch.cat([spp1, spp2, spp3], dim=1)

        # --- Metadata Branch ---
        # Ensure shape (Batch, 1)
        if x_meta.dim() == 1:
            x_meta = x_meta.unsqueeze(1)

        meta_feat = self.meta_fc1(x_meta)
        meta_feat = self.meta_bn1(meta_feat)
        meta_feat = self.meta_act(meta_feat)

        # --- Fusion ---
        combined = torch.cat([visual_feat, meta_feat], dim=1)

        out = self.fc1(combined)
        out = self.bn_fc1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)

        return torch.sigmoid(out)
