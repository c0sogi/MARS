import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class HybridSE(nn.Module):
    """
    Squeeze-and-Excitation Module using Global Average Pooling.
    Acts as a channel-wise attention mechanism robust to noise.
    """

    def __init__(self, channels, reduction=16):
        super(HybridSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Ensure reduction doesn't reduce channels below 1
        reduced_channels = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: Global Average Pooling
        y = self.avg_pool(x).view(b, c)
        # Excitation: Learn channel weights
        y = self.fc(y).view(b, c, 1, 1)
        # Scale
        return x * y


class ConvBlock(nn.Module):
    """
    Standard building block for the Plain CNN Backbone.
    Structure: Conv2d -> BN -> LeakyReLU -> SE -> MaxPool
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        # Bias is retained as per solution design
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=True
        )
        self.bn = nn.BatchNorm2d(out_channels)
        # LeakyReLU with negative slope 0.1 to preserve shadow information
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.se = HybridSE(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        x = self.pool(x)
        return x


class LSEIsomorphicCNN(nn.Module):
    """
    Log-Sum-Exp Isomorphic CNN (LSE-I-CNN).

    Features:
    - 4-Stage Plain CNN Backbone.
    - Selective Hierarchical Readout (Stage 3 & 4).
    - Decoupled 1x1 Projections.
    - Dual-Polarity Pooling:
        - Positive: Global Log-Sum-Exp (Total RCS)
        - Negative: Global SoftMin (Deepest Shadow)
    - Raw Incidence Angle Fusion.
    """

    def __init__(self):
        super(LSEIsomorphicCNN, self).__init__()

        # ---------------------------------------------------------------------
        # Backbone Construction
        # ---------------------------------------------------------------------
        self.stages = nn.ModuleList()
        in_c = Config.INPUT_CHANNELS

        # Build stages based on Config
        # Config.BACKBONE_CHANNELS = [64, 128, 128, 128]
        for out_c in Config.BACKBONE_CHANNELS:
            self.stages.append(ConvBlock(in_c, out_c))
            in_c = out_c

        # ---------------------------------------------------------------------
        # Readout Projections
        # ---------------------------------------------------------------------
        # We extract from Stage 3 (index 2) and Stage 4 (index 3)
        # Projections map backbone width to PROJECTION_DIM (64)

        # Projection for Stage 3 output
        self.proj3 = nn.Conv2d(
            Config.BACKBONE_CHANNELS[2], Config.PROJECTION_DIM, kernel_size=1, bias=True
        )

        # Projection for Stage 4 output
        self.proj4 = nn.Conv2d(
            Config.BACKBONE_CHANNELS[3], Config.PROJECTION_DIM, kernel_size=1, bias=True
        )

        # ---------------------------------------------------------------------
        # Classification Head
        # ---------------------------------------------------------------------
        # Feature Vector Size Calculation:
        # 2 Stages * 2 Polarities (LSE + SoftMin) * PROJECTION_DIM
        # 2 * 2 * 64 = 256
        self.feature_dim = len(Config.READOUT_STAGES) * 2 * Config.PROJECTION_DIM

        # Input to FC: Features + 1 (Incidence Angle)
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim + 1, Config.FC_DIM),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(p=Config.DROPOUT),
            nn.Linear(Config.FC_DIM, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """
        Kaiming Uniform Initialization for weights.
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
        """
        Args:
            x (torch.Tensor): Image input (B, 3, 75, 75)
            angle (torch.Tensor): Incidence angle input (B,)
        """
        features_map = {}

        # Pass through backbone stages
        out = x
        for i, stage in enumerate(self.stages):
            out = stage(out)
            # Store outputs of Stage 3 (idx 2) and Stage 4 (idx 3)
            if i in Config.READOUT_STAGES:
                features_map[i] = out

        # ---------------------------------------------------------------------
        # Isomorphic Dual-Polarity Pooling
        # ---------------------------------------------------------------------

        # Process Stage 3
        # 1. Project
        f3 = self.proj3(features_map[2])  # (B, 64, H3, W3)
        # 2. Positive Polarity: Global Log-Sum-Exp
        # log(sum(exp(x)))
        f3_pos = torch.logsumexp(f3, dim=(2, 3))  # (B, 64)
        # 3. Negative Polarity: Global SoftMin
        # -log(sum(exp(-x)))
        f3_neg = -torch.logsumexp(-f3, dim=(2, 3))  # (B, 64)

        # Process Stage 4
        # 1. Project
        f4 = self.proj4(features_map[3])  # (B, 64, H4, W4)
        # 2. Positive Polarity
        f4_pos = torch.logsumexp(f4, dim=(2, 3))  # (B, 64)
        # 3. Negative Polarity
        f4_neg = -torch.logsumexp(-f4, dim=(2, 3))  # (B, 64)

        # Concatenate all features
        # Order: [Stage3_Pos, Stage3_Neg, Stage4_Pos, Stage4_Neg]
        img_features = torch.cat([f3_pos, f3_neg, f4_pos, f4_neg], dim=1)  # (B, 256)

        # ---------------------------------------------------------------------
        # Fusion & Classification
        # ---------------------------------------------------------------------

        # Reshape angle to (B, 1)
        angle = angle.view(-1, 1)

        # Concatenate with raw incidence angle
        combined = torch.cat([img_features, angle], dim=1)  # (B, 257)

        # Head
        logits = self.head(combined)

        return logits
