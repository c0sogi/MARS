import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class HybridSE(nn.Module):
    """
    Hybrid Squeeze-and-Excitation Module.
    Uses Global Average Pooling for the squeeze operation to act as a low-pass filter.
    """

    def __init__(self, channels, reduction=16):
        super(HybridSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Ensure reduction doesn't make hidden dim < 1
        hidden_dim = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block for the Plain CNN backbone.
    Structure: Conv2d (bias=True) -> BatchNorm2d -> LeakyReLU -> SE -> MaxPool2d.
    """

    def __init__(self, in_channels, out_channels, negative_slope=0.1):
        super(ConvBlock, self).__init__()
        # Bias is explicitly retained as per instructions
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=True
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        self.se = HybridSE(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        x = self.pool(x)
        return x


class BDPH_CNN(nn.Module):
    """
    Bottlenecked Dual-Polarity Hierarchical CNN.

    Features:
    - 4-Stage Plain CNN Backbone.
    - Bottlenecked Dual-Polarity Readout from Stages 3 and 4.
    - Explicit handling of incidence angle without normalization.
    """

    def __init__(self):
        super(BDPH_CNN, self).__init__()

        # Hyperparameters from Config
        in_ch = Config.IN_CHANNELS
        widths = Config.BACKBONE_CHANNELS  # [64, 128, 128, 128]
        bottleneck_dim = Config.BOTTLENECK_DIM  # 32
        drop_rate = Config.DROPOUT_RATE
        slope = Config.LEAKY_RELU_SLOPE

        # --- Backbone ---
        # Stage 1: 75x75 -> 37x37
        self.block1 = ConvBlock(in_ch, widths[0], negative_slope=slope)
        # Stage 2: 37x37 -> 18x18
        self.block2 = ConvBlock(widths[0], widths[1], negative_slope=slope)
        # Stage 3: 18x18 -> 9x9
        self.block3 = ConvBlock(widths[1], widths[2], negative_slope=slope)
        # Stage 4: 9x9 -> 4x4
        self.block4 = ConvBlock(widths[2], widths[3], negative_slope=slope)

        # --- Bottlenecked Dual-Polarity Readout ---
        # Projects 128 channels down to 32 before pooling
        self.project_s3 = nn.Conv2d(widths[2], bottleneck_dim, kernel_size=1)
        self.project_s4 = nn.Conv2d(widths[3], bottleneck_dim, kernel_size=1)

        # --- Classification Head ---
        # Input features:
        #   Stage 3: 32 (Max) + 32 (Min) = 64
        #   Stage 4: 32 (Max) + 32 (Min) = 64
        #   Total Image Features: 128
        #   Incidence Angle: 1
        #   Total Dense Input: 129

        total_img_features = bottleneck_dim * 4
        dense_input_dim = total_img_features + 1

        self.head = nn.Sequential(
            nn.Linear(dense_input_dim, 256),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(256, 1),
        )

        # Initialization
        self._init_weights()

    def _init_weights(self):
        # PyTorch Default Initialization (Kaiming Uniform / Fan-In)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, inc_angle):
        # --- Backbone Forward ---
        x = self.block1(x)
        x = self.block2(x)

        feat_s3 = self.block3(x)  # Stage 3 Output (9x9)
        feat_s4 = self.block4(feat_s3)  # Stage 4 Output (4x4)

        # --- Bottlenecked Dual-Polarity Pooling ---

        # Process Stage 3
        p3 = self.project_s3(feat_s3)  # [B, 32, 9, 9]
        # Flatten spatial dims for pooling: [B, 32, 81]
        p3_flat = p3.view(p3.size(0), p3.size(1), -1)
        s3_max = p3_flat.max(dim=2)[0]  # Global Max
        s3_min = p3_flat.min(dim=2)[0]  # Global Min

        # Process Stage 4
        p4 = self.project_s4(feat_s4)  # [B, 32, 4, 4]
        # Flatten spatial dims for pooling: [B, 32, 16]
        p4_flat = p4.view(p4.size(0), p4.size(1), -1)
        s4_max = p4_flat.max(dim=2)[0]  # Global Max
        s4_min = p4_flat.min(dim=2)[0]  # Global Min

        # Aggregation: [B, 128]
        img_features = torch.cat([s3_max, s3_min, s4_max, s4_min], dim=1)

        # --- Feature Fusion ---
        # inc_angle is [B] or [B, 1]. Ensure it is [B, 1]
        if inc_angle.dim() == 1:
            inc_angle = inc_angle.unsqueeze(1)

        # Concatenate raw angle: [B, 129]
        combined = torch.cat([img_features, inc_angle], dim=1)

        # --- Head ---
        logits = self.head(combined)

        return logits
