import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class HybridSE(nn.Module):
    """
    Squeeze-and-Excitation Module.
    Uses Global Average Pooling (Squeeze) and a 2-layer FC MLP (Excitation)
    to recalibrate channel-wise feature responses.
    """

    def __init__(self, channels, reduction=16):
        super(HybridSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DPDModule(nn.Module):
    """
    Dual-Polarity Downsampling Module.
    Replaces standard MaxPool with parallel MaxPool (Peaks) and MinPool (Shadows).
    The results are concatenated and fused via a 1x1 Convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(DPDModule, self).__init__()
        self.pool_size = kernel_size
        self.stride = stride

        # Fusion layer: projects concatenated (in_channels * 2) to out_channels
        # Retains bias to preserve initialization dynamics
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(Config.LEAKY_RELU_SLOPE, inplace=True),
        )

    def forward(self, x):
        # Max Pooling (Captures Signal Peaks / Icebergs)
        p_max = F.max_pool2d(x, kernel_size=self.pool_size, stride=self.stride)

        # Min Pooling (Captures Signal Shadows / Voids)
        # Implemented as -MaxPool(-x)
        p_min = -F.max_pool2d(-x, kernel_size=self.pool_size, stride=self.stride)

        # Concatenate along channel dimension
        cat = torch.cat([p_max, p_min], dim=1)

        # Fuse back to target width
        out = self.fusion(cat)
        return out


class DPDCNN(nn.Module):
    """
    Dual-Polarity Downsampling CNN.
    4-Stage Plain CNN with DPD modules and Multi-Level Dual-Polarity Readout.
    """

    def __init__(self):
        super(DPDCNN, self).__init__()

        # Architecture Hyperparameters
        widths = Config.CHANNEL_WIDTHS  # [64, 128, 128, 128]
        slope = Config.LEAKY_RELU_SLOPE
        drop_rate = Config.DROPOUT_RATE

        # --- Stage 1 ---
        # Input: (B, 3, 75, 75) -> Output: (B, 128, 37, 37)
        self.s1_conv = nn.Conv2d(
            Config.IN_CHANNELS, widths[0], kernel_size=3, padding=1, bias=True
        )
        self.s1_bn = nn.BatchNorm2d(widths[0])
        self.s1_act = nn.LeakyReLU(slope, inplace=True)
        self.s1_se = HybridSE(widths[0])
        self.s1_dpd = DPDModule(widths[0], widths[1])  # 64 -> 128

        # --- Stage 2 ---
        # Input: (B, 128, 37, 37) -> Output: (B, 128, 18, 18)
        self.s2_conv = nn.Conv2d(
            widths[1], widths[1], kernel_size=3, padding=1, bias=True
        )
        self.s2_bn = nn.BatchNorm2d(widths[1])
        self.s2_act = nn.LeakyReLU(slope, inplace=True)
        self.s2_se = HybridSE(widths[1])
        self.s2_dpd = DPDModule(widths[1], widths[2])  # 128 -> 128

        # --- Stage 3 ---
        # Input: (B, 128, 18, 18) -> Output: (B, 128, 9, 9)
        self.s3_conv = nn.Conv2d(
            widths[2], widths[2], kernel_size=3, padding=1, bias=True
        )
        self.s3_bn = nn.BatchNorm2d(widths[2])
        self.s3_act = nn.LeakyReLU(slope, inplace=True)
        self.s3_se = HybridSE(widths[2])
        self.s3_dpd = DPDModule(widths[2], widths[3])  # 128 -> 128

        # --- Stage 4 ---
        # Input: (B, 128, 9, 9) -> Output: (B, 128, 9, 9)
        # No DPD here, feeds directly to readout
        self.s4_conv = nn.Conv2d(
            widths[3], widths[3], kernel_size=3, padding=1, bias=True
        )
        self.s4_bn = nn.BatchNorm2d(widths[3])
        self.s4_act = nn.LeakyReLU(slope, inplace=True)
        self.s4_se = HybridSE(widths[3])

        # --- Isomorphic Dual-Polarity Readout ---
        # Projects 128 channels to 64, then extracts both Max and Min stats.
        # Result per stage: 64 (max) + 64 (min) = 128 features.
        self.readout_proj_s3 = nn.Conv2d(widths[2], 64, kernel_size=1, bias=True)
        self.readout_proj_s4 = nn.Conv2d(widths[3], 64, kernel_size=1, bias=True)

        # --- Classification Head ---
        # Input: 128 (S3) + 128 (S4) + 1 (Angle) = 257
        self.head_fc1 = nn.Linear(257, 256, bias=True)
        self.head_act = nn.LeakyReLU(slope, inplace=True)
        self.head_drop = nn.Dropout(drop_rate)
        self.head_out = nn.Linear(256, 1, bias=True)

        self._init_weights()

    def _init_weights(self):
        """
        Kaiming Uniform Initialization (Fan-In) for stability.
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
        # --- Stage 1 ---
        x = self.s1_conv(x)
        x = self.s1_bn(x)
        x = self.s1_act(x)
        x = self.s1_se(x)
        x = self.s1_dpd(x)

        # --- Stage 2 ---
        x = self.s2_conv(x)
        x = self.s2_bn(x)
        x = self.s2_act(x)
        x = self.s2_se(x)
        x = self.s2_dpd(x)

        # --- Stage 3 ---
        x = self.s3_conv(x)
        x = self.s3_bn(x)
        x = self.s3_act(x)
        x = self.s3_se(x)
        feat_s3 = x  # Hook for readout
        x = self.s3_dpd(x)

        # --- Stage 4 ---
        x = self.s4_conv(x)
        x = self.s4_bn(x)
        x = self.s4_act(x)
        x = self.s4_se(x)
        feat_s4 = x  # Hook for readout

        # --- Readout Process ---
        # 1. Stage 3 Features
        p3 = self.readout_proj_s3(feat_s3)  # (B, 64, H, W)
        p3_max = F.adaptive_max_pool2d(p3, 1).view(p3.size(0), -1)  # (B, 64)
        p3_min = -F.adaptive_max_pool2d(-p3, 1).view(p3.size(0), -1)  # (B, 64)

        # 2. Stage 4 Features
        p4 = self.readout_proj_s4(feat_s4)  # (B, 64, H, W)
        p4_max = F.adaptive_max_pool2d(p4, 1).view(p4.size(0), -1)  # (B, 64)
        p4_min = -F.adaptive_max_pool2d(-p4, 1).view(p4.size(0), -1)  # (B, 64)

        # 3. Concatenate Features (128 + 128 = 256)
        features = torch.cat([p3_max, p3_min, p4_max, p4_min], dim=1)

        # --- Head ---
        # Concatenate Angle (256 + 1 = 257)
        angle = angle.view(-1, 1)
        combined = torch.cat([features, angle], dim=1)

        out = self.head_fc1(combined)
        out = self.head_act(out)
        out = self.head_drop(out)
        out = self.head_out(out)

        return out
