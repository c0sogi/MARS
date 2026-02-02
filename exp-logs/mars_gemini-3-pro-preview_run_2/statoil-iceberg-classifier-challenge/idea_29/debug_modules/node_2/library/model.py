import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config

# =============================================================================
# COMPONENTS
# =============================================================================


class DualPooling(nn.Module):
    """
    Applies Max Pooling and Min Pooling in parallel and concatenates the results.
    Doubles the number of channels.
    Min Pooling is implemented as -MaxPool(-x).
    """

    def __init__(self, kernel_size=2, stride=2):
        super(DualPooling, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        # Max Pooling (Peaks)
        out_max = self.max_pool(x)
        # Min Pooling (Shadows)
        out_min = -self.max_pool(-x)
        # Concatenate along channel dimension
        return torch.cat([out_max, out_min], dim=1)


class ChannelAttention(nn.Module):
    """
    CBAM Channel Attention Module.
    Uses Mixed Pooling (Avg + Max) and a shared MLP.
    """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        # Reduction ratio logic handled carefully for small channel counts
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
    CBAM Spatial Attention Module.
    Uses Channel-wise Avg and Max pooling concatenated, then 7x7 Conv.
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
    Sequentially applies Channel Attention then Spatial Attention.
    """

    def __init__(self, planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class DelayedIntegrationBlock(nn.Module):
    """
    A block that performs convolution, activation, and attention *before* pooling.
    Structure: Conv -> BN -> ReLU -> CBAM -> DualPooling
    """

    def __init__(self, in_channels, out_conv_channels):
        super(DelayedIntegrationBlock, self).__init__()

        # 1. Convolution
        self.conv = nn.Conv2d(
            in_channels, out_conv_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_conv_channels)
        self.relu = nn.ReLU(inplace=True)

        # 2. Pre-Pooling Attention
        self.cbam = CBAM(out_conv_channels)

        # 3. Dual-Stream Pooling (Output channels = 2 * out_conv_channels)
        self.dual_pool = DualPooling(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.cbam(x)
        x = self.dual_pool(x)
        return x


# =============================================================================
# MAIN MODEL: WB-DIN
# =============================================================================


class WBDIN(nn.Module):
    """
    Wide-Body Delayed-Integration Network.
    """

    def __init__(self):
        super(WBDIN, self).__init__()

        # --- Visual Branch (Backbone) ---
        # Stage 1: Input(3) -> Conv(64) -> DualPool(128)
        self.stage1 = DelayedIntegrationBlock(
            in_channels=config.N_CHANNELS, out_conv_channels=64
        )

        # Stage 2: Input(128) -> Conv(128) -> DualPool(256)
        self.stage2 = DelayedIntegrationBlock(in_channels=128, out_conv_channels=128)

        # Stage 3: Input(256) -> Conv(128) -> DualPool(256)
        self.stage3 = DelayedIntegrationBlock(in_channels=256, out_conv_channels=128)

        # Stage 4: Input(256) -> Conv(64) -> DualPool(128)
        self.stage4 = DelayedIntegrationBlock(in_channels=256, out_conv_channels=64)

        # --- Hybrid Readout Calculation ---
        # Input 75x75
        # S1 Pool -> 37x37
        # S2 Pool -> 18x18
        # S3 Pool -> 9x9
        # S4 Pool -> 4x4
        # Final channels: 128

        self.flat_dim = 4 * 4 * 128  # 2048
        self.peak_dim = 128
        self.visual_dim = self.flat_dim + self.peak_dim  # 2176

        # --- Metadata Branch ---
        self.use_metadata = config.USE_METADATA
        if self.use_metadata:
            self.meta_mlp = nn.Sequential(
                nn.Linear(config.METADATA_DIM, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 32),
                nn.ReLU(inplace=True),
            )
            self.meta_out_dim = 32
        else:
            self.meta_out_dim = 0

        # --- Fusion Head ---
        fusion_dim = self.visual_dim + self.meta_out_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(256, config.N_CLASSES),
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
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, inc_angle):
        # --- Visual Branch ---
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Hybrid Readout
        # Path A: Spatial Flatten
        x_flat = x.view(x.size(0), -1)

        # Path B: Global Max Pooling (Peak Intensity)
        x_peak = F.adaptive_max_pool2d(x, (1, 1)).view(x.size(0), -1)

        # Fusion
        visual_feat = torch.cat([x_flat, x_peak], dim=1)

        # --- Metadata Branch ---
        if self.use_metadata:
            # Ensure inc_angle is (Batch, 1)
            if inc_angle.dim() == 1:
                inc_angle = inc_angle.unsqueeze(1)
            meta_feat = self.meta_mlp(inc_angle)
            combined_feat = torch.cat([visual_feat, meta_feat], dim=1)
        else:
            combined_feat = visual_feat

        # --- Classifier ---
        logits = self.classifier(combined_feat)

        return logits
