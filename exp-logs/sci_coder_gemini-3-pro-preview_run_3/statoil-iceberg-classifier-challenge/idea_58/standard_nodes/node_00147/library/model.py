import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    INPUT_CHANNELS,
    BACKBONE_CHANNELS,
    SE_REDUCTION_RATIO,
    LEAKY_RELU_SLOPE,
    DROPOUT_RATE,
    USE_BIAS,
    NUM_CLASSES,
)


class SEBlock(nn.Module):
    """
    Standard Squeeze-and-Excitation Block.
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
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


class ConvBlock(nn.Module):
    """
    Convolutional Block: Conv -> BN -> LeakyReLU -> SE -> MaxPool.
    Explicitly retains bias in Conv2d.
    """

    def __init__(self, in_channels, out_channels, reduction=16, slope=0.1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=USE_BIAS
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=slope, inplace=True)
        self.se = SEBlock(out_channels, reduction=reduction)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        x = self.pool(x)
        return x


class CDICNN(nn.Module):
    """
    Corrected Decoupled Isomorphic CNN (CDI-CNN).

    Features:
    - Plain CNN Backbone (4 stages)
    - Stage-Decoupled Projections (Stage 3 & 4)
    - Isomorphic Pooling (Max + Min on same projection)
    - Physics-informed fusion (Concatenation with raw incidence angle)
    """

    def __init__(self):
        super(CDICNN, self).__init__()

        # --- Backbone ---
        # Stage 1: Input -> 64
        self.block1 = ConvBlock(
            INPUT_CHANNELS,
            BACKBONE_CHANNELS[0],
            reduction=SE_REDUCTION_RATIO,
            slope=LEAKY_RELU_SLOPE,
        )
        # Stage 2: 64 -> 128
        self.block2 = ConvBlock(
            BACKBONE_CHANNELS[0],
            BACKBONE_CHANNELS[1],
            reduction=SE_REDUCTION_RATIO,
            slope=LEAKY_RELU_SLOPE,
        )
        # Stage 3: 128 -> 128
        self.block3 = ConvBlock(
            BACKBONE_CHANNELS[1],
            BACKBONE_CHANNELS[2],
            reduction=SE_REDUCTION_RATIO,
            slope=LEAKY_RELU_SLOPE,
        )
        # Stage 4: 128 -> 128
        self.block4 = ConvBlock(
            BACKBONE_CHANNELS[2],
            BACKBONE_CHANNELS[3],
            reduction=SE_REDUCTION_RATIO,
            slope=LEAKY_RELU_SLOPE,
        )

        # --- Corrected Decoupled Isomorphic Readout ---
        # Decoupled Projections: Separate 1x1 convs for Stage 3 and 4
        # Reducing 128 channels to 64 before pooling
        self.proj3 = nn.Conv2d(BACKBONE_CHANNELS[2], 64, kernel_size=1, bias=USE_BIAS)
        self.proj4 = nn.Conv2d(BACKBONE_CHANNELS[3], 64, kernel_size=1, bias=USE_BIAS)

        # Feature Dimension Calculation:
        # Stage 3: 64 (Max) + 64 (Min) = 128
        # Stage 4: 64 (Max) + 64 (Min) = 128
        # Total Image Features = 256
        self.img_feature_dim = 128 + 128

        # Total Input to Classifier = Image Features + 1 (Incidence Angle)
        self.classifier_input_dim = self.img_feature_dim + 1

        # --- Classification Head ---
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_dim, 256),
            nn.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE, inplace=True),
            nn.Dropout(p=DROPOUT_RATE),
            nn.Linear(256, NUM_CLASSES),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        PyTorch Default Initialization (Kaiming Uniform/Fan-In).
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    m.weight,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                    a=LEAKY_RELU_SLOPE,
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                    a=LEAKY_RELU_SLOPE,
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, angle):
        # --- Backbone Forward ---
        x = self.block1(x)
        x = self.block2(x)

        # Stage 3
        x3 = self.block3(x)

        # Stage 4
        x4 = self.block4(x3)

        # --- Readout ---

        # Process Stage 3
        p3 = self.proj3(x3)  # (B, 64, H3, W3)
        # Global Max Pooling
        max3 = F.adaptive_max_pool2d(p3, (1, 1)).view(p3.size(0), -1)
        # Global Min Pooling (using negative max pool trick or view+min)
        # Using view+min for clarity and correctness
        min3 = p3.view(p3.size(0), p3.size(1), -1).min(dim=2)[0]

        # Process Stage 4
        p4 = self.proj4(x4)  # (B, 64, H4, W4)
        # Global Max Pooling
        max4 = F.adaptive_max_pool2d(p4, (1, 1)).view(p4.size(0), -1)
        # Global Min Pooling
        min4 = p4.view(p4.size(0), p4.size(1), -1).min(dim=2)[0]

        # Concatenate Isomorphic Features
        # Vector: [Max3, Min3, Max4, Min4]
        features = torch.cat([max3, min3, max4, min4], dim=1)  # (B, 256)

        # --- Fusion ---
        # Concatenate with raw incidence angle
        # Ensure angle is (B, 1)
        angle = angle.view(-1, 1)
        fused = torch.cat([features, angle], dim=1)  # (B, 257)

        # --- Classification ---
        out = self.classifier(fused)

        return out.squeeze(1)
