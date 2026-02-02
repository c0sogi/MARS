import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module for CBAM.
    Aggregates channel information using Global Average Pooling and Global Max Pooling.
    """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # Ensure hidden dimension is at least 1
        hidden_planes = max(1, in_planes // ratio)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
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
    Aggregates spatial information using Channel-wise Average and Max Pooling.
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


class ResidualIntegrationModule(nn.Module):
    """
    Residual Integration Module.
    Maps high-dimensional dual-stream input (256ch) to backbone width (128ch).
    Uses a residual structure to facilitate gradient flow.
    """

    def __init__(self, in_ch, out_ch):
        super(ResidualIntegrationModule, self).__init__()

        # Main Branch: 3x3 Conv -> BN -> ReLU
        self.main_conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.main_bn = nn.BatchNorm2d(out_ch)
        self.main_relu = nn.ReLU(inplace=True)

        # Shortcut Branch: 1x1 Conv -> BN (Projection)
        self.shortcut_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.shortcut_bn = nn.BatchNorm2d(out_ch)

        # Final Activation after addition
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Main Branch
        out = self.main_conv(x)
        out = self.main_bn(out)
        out = self.main_relu(out)

        # Shortcut Branch
        short = self.shortcut_conv(x)
        short = self.shortcut_bn(short)

        # Fusion
        out = out + short
        out = self.final_relu(out)
        return out


class DualPooling(nn.Module):
    """
    Dual-Stream Pooling.
    Applies Max Pooling (Peaks) and Min Pooling (Shadows) in parallel.
    Concatenates results, doubling the channel depth.
    """

    def __init__(self):
        super(DualPooling, self).__init__()
        self.max_pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        # Max Pooling for signal peaks
        x_max = self.max_pool(x)

        # Min Pooling for signal shadows (implemented as -Max(-x))
        x_min = -self.max_pool(-x)

        # Concatenate along channel dimension
        return torch.cat([x_max, x_min], dim=1)


class RIWBN(nn.Module):
    """
    Residual-Integrated Wide-Body Network.
    """

    def __init__(self):
        super(RIWBN, self).__init__()

        # --- Stage 1 (Stem) ---
        # Input: 3 channels (Band1, Band2, Mean)
        # Map directly to 128 channels to establish wide backbone immediately
        self.stage1_conv = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.stage1_bn = nn.BatchNorm2d(128)
        self.stage1_relu = nn.ReLU(inplace=True)
        self.stage1_cbam = CBAM(128)
        self.stage1_pool = DualPooling()  # Output: 128 -> 256 channels, 75->37 spatial

        # --- Stage 2 ---
        # Input: 256 channels (from DualPool)
        self.stage2_integ = ResidualIntegrationModule(256, 128)
        self.stage2_cbam = CBAM(128)
        self.stage2_pool = DualPooling()  # Output: 128 -> 256 channels, 37->18 spatial

        # --- Stage 3 ---
        self.stage3_integ = ResidualIntegrationModule(256, 128)
        self.stage3_cbam = CBAM(128)
        self.stage3_pool = DualPooling()  # Output: 128 -> 256 channels, 18->9 spatial

        # --- Stage 4 ---
        self.stage4_integ = ResidualIntegrationModule(256, 128)
        self.stage4_cbam = CBAM(128)
        self.stage4_pool = DualPooling()  # Output: 128 -> 256 channels, 9->4 spatial

        # --- Dual-Path Readout ---
        # Input: 256 channels, 4x4 spatial

        # Path A: Spatial Context
        # Compresses channels but keeps spatial dims for flattening
        self.path_a_conv = nn.Conv2d(256, 48, kernel_size=3, padding=1, bias=False)

        # Path B: Robust Intensity
        # Global Average Pooling for statistical robustness
        self.path_b_pool = nn.AdaptiveAvgPool2d(1)

        # --- Metadata Branch ---
        self.meta_fc = nn.Sequential(nn.Linear(1, 16), nn.ReLU(inplace=True))

        # --- Fusion Head ---
        # Path A size: 4x4 spatial * 48 channels = 768
        # Path B size: 256 channels
        # Meta size: 16
        # Total input: 768 + 256 + 16 = 1040
        self.classifier = nn.Sequential(
            nn.Linear(1040, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )

        # Weight Initialization
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

    def forward(self, x, inc_angle):
        # Stage 1
        x = self.stage1_conv(x)
        x = self.stage1_bn(x)
        x = self.stage1_relu(x)
        x = self.stage1_cbam(x)
        x = self.stage1_pool(x)

        # Stage 2
        x = self.stage2_integ(x)
        x = self.stage2_cbam(x)
        x = self.stage2_pool(x)

        # Stage 3
        x = self.stage3_integ(x)
        x = self.stage3_cbam(x)
        x = self.stage3_pool(x)

        # Stage 4
        x = self.stage4_integ(x)
        x = self.stage4_cbam(x)
        x = self.stage4_pool(x)

        # Readout
        # Path A: Spatial (Flattened Conv)
        xa = self.path_a_conv(x)
        xa = xa.view(xa.size(0), -1)  # Flatten (B, 768)

        # Path B: Intensity (Global Avg Pool)
        xb = self.path_b_pool(x)
        xb = xb.view(xb.size(0), -1)  # Flatten (B, 256)

        # Metadata
        xm = inc_angle.view(-1, 1)  # Ensure (B, 1)
        xm = self.meta_fc(xm)

        # Fusion
        feat = torch.cat([xa, xb, xm], dim=1)
        out = self.classifier(feat)

        return out
