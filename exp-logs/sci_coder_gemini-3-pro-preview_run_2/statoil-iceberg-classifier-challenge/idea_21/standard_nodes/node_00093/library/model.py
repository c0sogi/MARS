import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DualPooling(nn.Module):
    """
    Dual-Stream Pooling: Concatenates Max Pooling and Min Pooling.
    Doubles the number of channels.
    """

    def __init__(self, kernel_size=2, stride=2, padding=0):
        super(DualPooling, self).__init__()
        self.max_pool = nn.MaxPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        # Min pooling is implemented as -max_pool(-x)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # Max Pool
        out_max = self.max_pool(x)

        # Min Pool
        out_min = -F.max_pool2d(
            -x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
        )

        # Concatenate along channel dimension
        return torch.cat([out_max, out_min], dim=1)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        hidden_channels = max(in_channels // reduction_ratio, 8)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
        avg_out = self.avg_pool(x).flatten(1)
        max_out = self.max_pool(x).flatten(1)

        # Apply shared MLP
        avg_out = self.mlp(avg_out)
        max_out = self.mlp(max_out)

        # Sum and Sigmoid
        out = self.sigmoid(avg_out + max_out)

        # Reshape for broadcasting: (B, C, 1, 1)
        return out.unsqueeze(2).unsqueeze(3)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Input will be 2 channels (Max + Avg)
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise Avg and Max pooling -> (B, 1, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concat -> (B, 2, H, W)
        x_cat = torch.cat([avg_out, max_out], dim=1)

        # Conv -> Sigmoid
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Applied strictly before pooling.
    """

    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x):
        # Channel Attention
        out = x * self.channel_att(x)
        # Spatial Attention
        out = out * self.spatial_att(out)
        return out


class ConvBlock(nn.Module):
    """
    Standard block: Conv -> BN -> ReLU -> CBAM -> DualPool
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        # Convolution
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Attention
        self.cbam = CBAM(out_channels)

        # Dual Pooling (doubles output channels relative to out_channels)
        self.dual_pool = DualPooling(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.cbam(x)
        x = self.dual_pool(x)
        return x


class QPWBN(nn.Module):
    """
    Contracting Dual-Pooling Network (formerly QPWBN).
    Implements contracting channel strategy and direct flattening.
    """

    def __init__(self):
        super(QPWBN, self).__init__()

        # 1. Visual Branch (Contracting Dual-Pooling Backbone)
        self.features = nn.Sequential()

        in_c = Config.IN_CHANNELS
        filter_sizes = Config.FILTER_SIZES  # [64, 64, 64, 64]

        # We need to track the actual channel count flowing between blocks
        # because DualPooling doubles the channel count at the output of each block.
        current_channels = in_c

        for i, out_c in enumerate(filter_sizes):
            self.features.add_module(f"block_{i+1}", ConvBlock(current_channels, out_c))
            # The input to the next layer is out_c * 2 (due to DualPooling)
            current_channels = out_c * 2

        # Final visual channels: 64 * 2 = 128
        self.visual_out_channels = current_channels

        # Readout
        # Direct flattening of 4x4 spatial grid (Cite Lesson 21, Lesson 43)
        # 75 -> 37 -> 18 -> 9 -> 4
        self.spatial_dim = 4
        self.visual_flat_dim = self.visual_out_channels * (self.spatial_dim**2)

        # 2. Metadata Branch (MLP for inc_angle)
        self.meta_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.meta_out_dim = 32

        # 3. Fusion Head
        fusion_dim = self.visual_flat_dim + self.meta_out_dim  # 1024 + 32 = 1056

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=Config.DROPOUT_RATE),
            nn.Linear(512, 1),
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
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_img, x_meta):
        # Visual Branch
        v = self.features(x_img)
        v = v.flatten(1)  # Direct flatten (Cite Lesson 43)

        # Metadata Branch (ensure shape (B, 1))
        if x_meta.dim() == 1:
            x_meta = x_meta.unsqueeze(1)
        m = self.meta_mlp(x_meta)

        # Fusion
        combined = torch.cat([v, m], dim=1)
        logits = self.classifier(combined)

        return logits.squeeze(1)
