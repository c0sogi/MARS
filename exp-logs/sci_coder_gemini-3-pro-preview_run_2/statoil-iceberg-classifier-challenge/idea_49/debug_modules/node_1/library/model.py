import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # Ensure hidden planes is at least a reasonable size
        hidden_planes = max(4, in_planes // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

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
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        # Input channels = 2 (Max + Avg)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Mixed Pooling (Max + Avg)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
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
    Dual-Stream Pooling: Concatenates Max Pooling (Peaks) and Min Pooling (Shadows).
    Min Pooling is implemented as -MaxPool(-x).
    """

    def __init__(self):
        super(DualPooling, self).__init__()
        self.max_pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        # Max Pooling (Peaks)
        x_max = self.max_pool(x)
        # Min Pooling (Shadows)
        x_min = -self.max_pool(-x)
        # Concatenate: Input C -> Output 2C
        return torch.cat([x_max, x_min], dim=1)


class DCSWBN(nn.Module):
    """
    Decoupled Contrast-Shape Wide-Body Network (DCS-WBN).
    """

    def __init__(self):
        super(DCSWBN, self).__init__()

        # Configuration
        base_filters = Config.BASE_FILTERS  # 128

        # --- Visual Branch (Wide-Body Delayed-Integration Backbone) ---

        # Stage 1 (Stem)
        # Input: 3 channels (Bands + Mean) -> 128 filters
        self.stage1_conv = nn.Conv2d(
            Config.IN_CHANNELS, base_filters, kernel_size=3, padding=1
        )
        self.stage1_bn = nn.BatchNorm2d(base_filters)
        self.stage1_relu = nn.ReLU(inplace=True)
        self.stage1_cbam = CBAM(base_filters)
        self.stage1_pool = DualPooling()  # Output channels: 128 * 2 = 256

        # Stage 2
        # Input: 256 -> 128 (Delayed Integration)
        self.stage2_conv = nn.Conv2d(
            base_filters * 2, base_filters, kernel_size=3, padding=1
        )
        self.stage2_bn = nn.BatchNorm2d(base_filters)
        self.stage2_relu = nn.ReLU(inplace=True)
        self.stage2_cbam = CBAM(base_filters)
        self.stage2_pool = DualPooling()  # Output channels: 256

        # Stage 3
        # Input: 256 -> 128
        self.stage3_conv = nn.Conv2d(
            base_filters * 2, base_filters, kernel_size=3, padding=1
        )
        self.stage3_bn = nn.BatchNorm2d(base_filters)
        self.stage3_relu = nn.ReLU(inplace=True)
        self.stage3_cbam = CBAM(base_filters)
        self.stage3_pool = DualPooling()  # Output channels: 256

        # Stage 4
        # Input: 256 -> 128
        self.stage4_conv = nn.Conv2d(
            base_filters * 2, base_filters, kernel_size=3, padding=1
        )
        self.stage4_bn = nn.BatchNorm2d(base_filters)
        self.stage4_relu = nn.ReLU(inplace=True)
        self.stage4_cbam = CBAM(base_filters)
        self.stage4_pool = DualPooling()  # Output channels: 256

        # --- Decoupled Readout ---

        # Path A: Shape (Contrast Invariant)
        # Input: 256 channels. Conv -> 64.
        # Instance Normalization forces learning of shape regardless of global intensity.
        self.path_a_conv = nn.Conv2d(base_filters * 2, 64, kernel_size=3, padding=1)
        self.path_a_in = nn.InstanceNorm2d(64, affine=True)
        # Output size at stage 4 is 4x4 (75 -> 37 -> 18 -> 9 -> 4).
        # Flatten dim: 64 * 4 * 4 = 1024
        self.path_a_dim = 64 * 4 * 4

        # Path B: Intensity (Absolute Signal)
        # Input: 256 channels. GAP -> 256.
        # Batch Normalization preserves relative intensity distributions across the batch.
        self.path_b_gap = nn.AdaptiveAvgPool2d(1)
        self.path_b_bn = nn.BatchNorm1d(base_filters * 2)
        self.path_b_dim = base_filters * 2

        # --- Metadata Branch ---
        # Structure: Linear -> ReLU -> Linear -> BN -> ReLU
        self.meta_fc1 = nn.Linear(1, 16)
        self.meta_relu1 = nn.ReLU(inplace=True)
        self.meta_fc2 = nn.Linear(16, 32)
        self.meta_bn = nn.BatchNorm1d(32)
        self.meta_relu2 = nn.ReLU(inplace=True)
        self.meta_dim = 32

        # --- Fusion Head ---
        fusion_dim = (
            self.path_a_dim + self.path_b_dim + self.meta_dim
        )  # 1024 + 256 + 32 = 1312

        self.fusion_fc = nn.Linear(fusion_dim, 512)
        self.fusion_bn = nn.BatchNorm1d(512)
        self.fusion_relu = nn.ReLU(inplace=True)
        self.fusion_dropout = nn.Dropout(Config.DROPOUT_RATE)
        self.output_fc = nn.Linear(512, 1)

    def forward(self, x_img, x_meta):
        # x_img: (B, 3, 75, 75)
        # x_meta: (B, 1) or (B,)

        # Ensure x_meta is (B, 1)
        if x_meta.dim() == 1:
            x_meta = x_meta.unsqueeze(1)

        # --- Visual Branch ---
        # Stage 1
        x = self.stage1_conv(x_img)
        x = self.stage1_bn(x)
        x = self.stage1_relu(x)
        x = self.stage1_cbam(x)
        x = self.stage1_pool(x)

        # Stage 2
        x = self.stage2_conv(x)
        x = self.stage2_bn(x)
        x = self.stage2_relu(x)
        x = self.stage2_cbam(x)
        x = self.stage2_pool(x)

        # Stage 3
        x = self.stage3_conv(x)
        x = self.stage3_bn(x)
        x = self.stage3_relu(x)
        x = self.stage3_cbam(x)
        x = self.stage3_pool(x)

        # Stage 4
        x = self.stage4_conv(x)
        x = self.stage4_bn(x)
        x = self.stage4_relu(x)
        x = self.stage4_cbam(x)
        x = self.stage4_pool(x)  # (B, 256, 4, 4)

        # --- Decoupled Readout ---

        # Path A (Shape)
        xa = self.path_a_conv(x)
        xa = self.path_a_in(xa)
        xa = xa.view(xa.size(0), -1)  # Flatten

        # Path B (Intensity)
        xb = self.path_b_gap(x)
        xb = xb.view(xb.size(0), -1)
        xb = self.path_b_bn(xb)

        # --- Metadata Branch ---
        xm = self.meta_fc1(x_meta)
        xm = self.meta_relu1(xm)
        xm = self.meta_fc2(xm)
        xm = self.meta_bn(xm)
        xm = self.meta_relu2(xm)

        # --- Fusion ---
        fused = torch.cat([xa, xb, xm], dim=1)

        out = self.fusion_fc(fused)
        out = self.fusion_bn(out)
        out = self.fusion_relu(out)
        out = self.fusion_dropout(out)
        out = self.output_fc(out)

        return out
