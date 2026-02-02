import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DPSCAModule(nn.Module):
    """
    Dual-Polarity Spatial-Channel Attention Module.
    Combines 'Soft' Dual-Polarity Spatial Attention with Global Channel Attention.
    """

    def __init__(self, channels, reduction=16):
        super(DPSCAModule, self).__init__()

        # --- Channel Attention ---
        # Global Average Pooling acts as a low-pass filter
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Squeeze-Excitation style MLP
        # Ensure reduction doesn't reduce channels below a reasonable limit
        mid_channels = max(channels // reduction, 4)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid(),
        )

        # --- Spatial Attention (Dual-Polarity) ---
        # Input to conv is 2 channels: Channel-wise Max + Channel-wise Min
        # 7x7 Convolution to capture local context around peaks/shadows
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. Channel Attention
        b, c, _, _ = x.size()
        y_c = self.avg_pool(x).view(b, c)
        y_c = self.channel_mlp(y_c).view(b, c, 1, 1)
        x = x * y_c

        # 2. Spatial Attention
        # Channel-wise Max Pooling (Strongest Signal)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        # Channel-wise Min Pooling (Deepest Shadow)
        min_pool, _ = torch.min(x, dim=1, keepdim=True)

        # Concatenate to form Dual-Polarity map
        y_s = torch.cat([max_pool, min_pool], dim=1)

        # Convolve and Sigmoid
        y_s = self.spatial_conv(y_s)
        y_s = self.spatial_sigmoid(y_s)

        # Apply Spatial Mask
        x = x * y_s
        return x


class ConvBlock(nn.Module):
    """
    Standard CNN Block with DPSCA integration.
    Conv2d -> BN -> LeakyReLU -> DPSCA -> MaxPool
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        # Explicitly retain bias to preserve initialization dynamics
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=True
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(Config.LEAKY_RELU_SLOPE, inplace=True)
        self.att = DPSCAModule(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.att(x)
        x = self.pool(x)
        return x


class DPSCACNN(nn.Module):
    """
    Dual-Polarity Spatial-Channel Attention CNN.
    A Custom 4-Stage Attentive Convolutional Network.
    """

    def __init__(self):
        super(DPSCACNN, self).__init__()

        # Backbone Configuration
        # Strategy: 64 -> 128 -> 128 -> 128 (Early Expansion)
        widths = Config.BACKBONE_CHANNELS

        self.block1 = ConvBlock(Config.IN_CHANNELS, widths[0])
        self.block2 = ConvBlock(widths[0], widths[1])
        self.block3 = ConvBlock(widths[1], widths[2])
        self.block4 = ConvBlock(widths[2], widths[3])

        # Readout Compression (128 -> 64)
        # Applied to Stage 3 and Stage 4 outputs
        self.compress3 = nn.Conv2d(widths[2], 64, kernel_size=1)
        self.compress4 = nn.Conv2d(widths[3], 64, kernel_size=1)

        # Classification Head
        # Feature Vector Calculation:
        # Stage 3: 64 ch (Max) + 64 ch (Min) = 128
        # Stage 4: 64 ch (Max) + 64 ch (Min) = 128
        # Total Features = 256
        # + Incidence Angle = 257
        input_dim = 257
        hidden_dim = Config.FC_DIM  # 256

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(Config.LEAKY_RELU_SLOPE, inplace=True),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(hidden_dim, Config.NUM_CLASSES),
        )

        self._init_weights()

    def _init_weights(self):
        """
        Kaiming Uniform Initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, angle):
        # --- Backbone ---
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)  # Stage 3 Output
        x4 = self.block4(x3)  # Stage 4 Output

        # --- Isomorphic Dual-Polarity Readout ---

        # Stage 3 Processing
        c3 = self.compress3(x3)
        # Global Max Pooling
        gmp3 = F.adaptive_max_pool2d(c3, 1).view(c3.size(0), -1)
        # Global Min Pooling
        b, c, h, w = c3.size()
        c3_flat = c3.view(b, c, -1)
        gmin3 = torch.min(c3_flat, dim=2)[0]

        # Stage 4 Processing
        c4 = self.compress4(x4)
        # Global Max Pooling
        gmp4 = F.adaptive_max_pool2d(c4, 1).view(c4.size(0), -1)
        # Global Min Pooling
        b, c, h, w = c4.size()
        c4_flat = c4.view(b, c, -1)
        gmin4 = torch.min(c4_flat, dim=2)[0]

        # Concatenate all visual features
        # Size: B x 256
        features = torch.cat([gmp3, gmin3, gmp4, gmin4], dim=1)

        # --- Fusion ---
        # Ensure angle is (B, 1) and float
        if angle.dim() == 1:
            angle = angle.view(-1, 1)
        angle = angle.float()

        # Size: B x 257
        final_vec = torch.cat([features, angle], dim=1)

        # --- Classification Head ---
        out = self.fc(final_vec)
        return out
