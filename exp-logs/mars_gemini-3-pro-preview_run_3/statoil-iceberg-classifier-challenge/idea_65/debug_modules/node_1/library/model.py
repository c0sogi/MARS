import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class HybridSE(nn.Module):
    """
    Hybrid Squeeze-and-Excitation Module.
    Uses Global Average Pooling for the squeeze operation to be robust to noise.
    """

    def __init__(self, channels, reduction=16):
        super(HybridSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Scale
        return x * y


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block for the Plain CNN Backbone.
    Structure: Conv2d(bias=True) -> BN -> LeakyReLU -> HybridSE -> MaxPool.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, slope=0.1):
        super(ConvBlock, self).__init__()
        padding = kernel_size // 2

        # Explicitly retaining bias=True as per design
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=slope, inplace=True)
        self.se = HybridSE(out_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        x = self.pool(x)
        return x


class CMSDI_CNN(nn.Module):
    """
    Corrected Multi-Sample Dropout Isomorphic CNN (CMSDI-CNN).

    Features:
    - 4-Stage Plain CNN Backbone (Aggressive Downsampling).
    - Selective Hierarchical Readout (Stage 3 & 4).
    - Decoupled Isomorphic Pooling (Max + Min).
    - Raw Incidence Angle Fusion.
    - Non-Linear Interaction Layer (Correction from Idea 64).
    - Multi-Sample Dropout Head.
    """

    def __init__(self):
        super(CMSDI_CNN, self).__init__()

        # Hyperparameters from Config
        self.channels = Config.BACKBONE_CHANNELS  # [64, 128, 128, 128]
        self.slope = Config.LEAKY_RELU_SLOPE
        self.hidden_dim = Config.HIDDEN_DIM
        self.dropout_rate = Config.DROPOUT_RATE
        self.num_samples = Config.NUM_DROPOUT_SAMPLES

        # --- Backbone ---
        # Input: 3 channels (HH, HV, Avg)
        self.stage1 = ConvBlock(
            Config.NUM_INPUT_CHANNELS, self.channels[0], slope=self.slope
        )
        self.stage2 = ConvBlock(self.channels[0], self.channels[1], slope=self.slope)
        self.stage3 = ConvBlock(self.channels[1], self.channels[2], slope=self.slope)
        self.stage4 = ConvBlock(self.channels[2], self.channels[3], slope=self.slope)

        # --- Readout (Decoupled Isomorphic) ---
        # Stage 3 Projection: 128 -> 64
        self.proj3 = nn.Conv2d(self.channels[2], 64, kernel_size=1)
        # Stage 4 Projection: 128 -> 64
        self.proj4 = nn.Conv2d(self.channels[3], 64, kernel_size=1)

        # Visual Vector Size:
        # (Stage3 Max 64 + Stage3 Min 64) + (Stage4 Max 64 + Stage4 Min 64) = 256
        self.visual_dim = 64 * 4

        # --- Interaction Layer ---
        # Input: Visual Vector (256) + Incidence Angle (1) = 257
        self.fusion_dim = self.visual_dim + 1
        self.interaction = nn.Sequential(
            nn.Linear(self.fusion_dim, self.hidden_dim),
            nn.LeakyReLU(negative_slope=self.slope, inplace=True),
        )

        # --- Multi-Sample Dropout Head ---
        # 5 parallel linear layers
        self.head_projections = nn.ModuleList(
            [nn.Linear(self.hidden_dim, 1) for _ in range(self.num_samples)]
        )
        self.dropout = nn.Dropout(p=self.dropout_rate)

        # Initialization
        self._init_weights()

    def _init_weights(self):
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

    def forward_features(self, x):
        # Backbone
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)  # Keep for readout
        x4 = self.stage4(x3)  # Keep for readout

        # Readout Stage 3
        p3 = self.proj3(x3)
        # Global Max Pooling
        max3 = F.adaptive_max_pool2d(p3, 1).view(p3.size(0), -1)
        # Global Min Pooling (implemented as -max(-x))
        min3 = -F.adaptive_max_pool2d(-p3, 1).view(p3.size(0), -1)

        # Readout Stage 4
        p4 = self.proj4(x4)
        max4 = F.adaptive_max_pool2d(p4, 1).view(p4.size(0), -1)
        min4 = -F.adaptive_max_pool2d(-p4, 1).view(p4.size(0), -1)

        # Concatenate all visual features
        visual_feat = torch.cat([max3, min3, max4, min4], dim=1)
        return visual_feat

    def forward(self, x, angle):
        """
        Args:
            x: Image tensor [Batch, 3, 75, 75]
            angle: Incidence angle tensor [Batch] or [Batch, 1]
        """
        # Ensure angle is [Batch, 1]
        if angle.dim() == 1:
            angle = angle.view(-1, 1)

        # 1. Feature Extraction
        visual_feat = self.forward_features(x)

        # 2. Fusion
        # Concatenate raw angle
        fused = torch.cat([visual_feat, angle], dim=1)

        # 3. Interaction (Non-Linear Hidden Layer)
        # This learns the joint embedding before dropout
        embedding = self.interaction(fused)

        # 4. Multi-Sample Dropout Head
        logits_list = []
        for i in range(self.num_samples):
            # Apply dropout to the shared embedding
            d = self.dropout(embedding)
            # Apply specific projection
            logits_list.append(self.head_projections[i](d))

        # Stack logits: [Batch, Num_Samples]
        stacked_logits = torch.cat(logits_list, dim=1)

        if self.training:
            # During training, return all logits for multi-sample loss
            return stacked_logits
        else:
            # During inference, average the logits
            return torch.mean(stacked_logits, dim=1, keepdim=True)
