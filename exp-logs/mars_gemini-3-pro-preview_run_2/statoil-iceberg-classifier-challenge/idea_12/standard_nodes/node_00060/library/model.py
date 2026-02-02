import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


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

        # Ensure hidden dimension is at least 1
        hidden_planes = max(in_planes // ratio, 1)

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
    Aggregates channel information using Max and Average pooling along the channel axis,
    then uses a convolution to generate a spatial attention map.
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
    Sequentially applies Channel Attention and Spatial Attention.
    """

    def __init__(self, planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block for the GLPPN Backbone.
    Structure: Conv -> BN -> ReLU -> MaxPool -> CBAM
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cbam = CBAM(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.cbam(x)
        return x


class A2SHN(nn.Module):
    """
    Attention-Augmented Shallow Hybrid Network (A2SHN).

    Optimizations:
    - Cite solution_lesson_node_00041: Uses channel contraction in the final block (128 -> 64)
      to reduce the dimensionality of the flattened vector.
    - Cite solution_lesson_node_00039: Avoids multi-scale/dilated branches for small 75x75 inputs.
    - Cite solution_lesson_node_00043: Uses simple flattening instead of complex pooling.
    """

    def __init__(self):
        super(A2SHN, self).__init__()

        # --- Visual Branch (Contracted-Width Backbone) ---
        filters = Config.BACKBONE_FILTERS  # [64, 128, 128, 64]
        input_channels = Config.CHANNELS  # 3

        self.layer1 = ConvBlock(input_channels, filters[0])
        self.layer2 = ConvBlock(filters[0], filters[1])
        self.layer3 = ConvBlock(filters[1], filters[2])
        self.layer4 = ConvBlock(filters[2], filters[3])

        # --- Aggregation Head ---
        # Based on 75x75 input:
        # L1 -> 37x37
        # L2 -> 18x18
        # L3 -> 9x9
        # L4 -> 4x4
        self.feature_h = 4
        self.feature_w = 4
        self.final_filters = filters[3]  # 64

        self.flat_dim = (
            self.final_filters * self.feature_h * self.feature_w
        )  # 64 * 4 * 4 = 1024

        # --- Metadata Branch ---
        self.meta_fc = nn.Sequential(
            nn.Linear(1, Config.META_HIDDEN_DIM),
            nn.BatchNorm1d(Config.META_HIDDEN_DIM),
            nn.ReLU(inplace=True),
        )

        # --- Classification Head ---
        # Fusion: Flattened Visual + Metadata
        fusion_dim = self.flat_dim + Config.META_HIDDEN_DIM

        # Dense -> BatchNorm -> ReLU -> Dropout -> Output
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(512, 1),  # Output logits
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, angle):
        """
        Args:
            x (torch.Tensor): Image batch of shape (B, 3, 75, 75)
            angle (torch.Tensor): Incidence angle batch of shape (B, 1)
        """
        # 1. Visual Backbone
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # (B, 64, 4, 4)

        # 2. Aggregation Head (Flatten)
        # Cite solution_lesson_node_00043: Simple flattening preserves spatial structure.
        flat = x.view(x.size(0), -1)  # (B, 1024)

        # 3. Metadata Branch
        meta = self.meta_fc(angle)  # (B, 32)

        # 4. Fusion
        fused = torch.cat((flat, meta), dim=1)

        # 5. Classification
        out = self.classifier(fused)

        return out
