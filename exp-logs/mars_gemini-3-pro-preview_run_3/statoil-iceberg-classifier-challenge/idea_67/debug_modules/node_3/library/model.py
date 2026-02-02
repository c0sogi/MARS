import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    BACKBONE_CHANNELS,
    SE_REDUCTION,
    LEAKY_RELU_SLOPE,
    READOUT_PROJ_DIM,
    CLASSIFIER_HIDDEN_DIM,
    DROPOUT_RATE,
    INPUT_CHANNELS,
)


class LeakySEModule(nn.Module):
    """
    Leaky Squeeze-and-Excitation Module.

    Replaces the standard ReLU in the SE bottleneck with LeakyReLU to allow
    attention weights to be derived from negative signal components (shadows).
    """

    def __init__(self, channels, reduction=SE_REDUCTION, slope=LEAKY_RELU_SLOPE):
        super(LeakySEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(1, channels // reduction)

        self.fc1 = nn.Linear(channels, reduced_channels, bias=True)
        self.act = nn.LeakyReLU(negative_slope=slope, inplace=True)
        self.fc2 = nn.Linear(reduced_channels, channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: Global Average Pooling
        y = self.avg_pool(x).view(b, c)

        # Excitation: Linear -> LeakyReLU -> Linear -> Sigmoid
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)

        # Scale
        return x * y


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block for the Plain CNN Backbone.
    Structure: Conv2d -> BN -> LeakyReLU -> LeakySE -> MaxPool
    """

    def __init__(self, in_channels, out_channels, slope=LEAKY_RELU_SLOPE):
        super(ConvBlock, self).__init__()
        # Bias is retained to preserve initialization dynamics
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=True
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=slope, inplace=True)
        self.se = LeakySEModule(out_channels, slope=slope)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        x = self.pool(x)
        return x


class LeakyAttentiveIsomorphicCNN(nn.Module):
    """
    Idea 67: Leaky-Attentive Isomorphic CNN (LA-I-CNN).

    Combines a Plain CNN backbone with Leaky-SE modules and a Corrected
    Decoupled Isomorphic Readout to handle SAR signal characteristics
    (shadows/peaks) and incidence angle fusion.
    """

    def __init__(self):
        super(LeakyAttentiveIsomorphicCNN, self).__init__()

        # --- Backbone ---
        # Stage 1: Input -> 64
        self.stage1 = ConvBlock(INPUT_CHANNELS, BACKBONE_CHANNELS[0])

        # Stage 2: 64 -> 128
        self.stage2 = ConvBlock(BACKBONE_CHANNELS[0], BACKBONE_CHANNELS[1])

        # Stage 3: 128 -> 128
        self.stage3 = ConvBlock(BACKBONE_CHANNELS[1], BACKBONE_CHANNELS[2])

        # Stage 4: 128 -> 128
        self.stage4 = ConvBlock(BACKBONE_CHANNELS[2], BACKBONE_CHANNELS[3])

        # --- Readout (Corrected Decoupled Isomorphic) ---
        # Projections for Stage 3 and Stage 4
        self.proj3 = nn.Conv2d(
            BACKBONE_CHANNELS[2], READOUT_PROJ_DIM, kernel_size=1, bias=True
        )
        self.proj4 = nn.Conv2d(
            BACKBONE_CHANNELS[3], READOUT_PROJ_DIM, kernel_size=1, bias=True
        )

        # --- Classifier Head ---
        # Input dim calculation:
        # Stage 3: Max(64) + Min(64) = 128
        # Stage 4: Max(64) + Min(64) = 128
        # Total Image Features: 256
        # Incidence Angle: 1
        # Total Input: 257

        self.head_input_dim = (READOUT_PROJ_DIM * 4) + 1

        self.classifier = nn.Sequential(
            nn.Linear(self.head_input_dim, CLASSIFIER_HIDDEN_DIM),
            nn.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE, inplace=True),
            nn.Dropout(p=DROPOUT_RATE),
            nn.Linear(CLASSIFIER_HIDDEN_DIM, 1),
        )

        # Initialization: PyTorch default initialization is used implicitly.
        # No explicit init_weights function is called to strictly adhere to "PyTorch Default".

    def forward_features(self, x):
        """
        Forward pass through the backbone and readout.
        Returns the flattened feature vector.
        """
        # Backbone Stages
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)  # Feature Map from Stage 3
        x4 = self.stage4(x3)  # Feature Map from Stage 4

        # Decoupled Projections
        feat3 = self.proj3(x3)
        feat4 = self.proj4(x4)

        # Isomorphic Pooling (Max + Min)
        # Global Max Pooling
        max3 = F.adaptive_max_pool2d(feat3, (1, 1)).view(feat3.size(0), -1)
        max4 = F.adaptive_max_pool2d(feat4, (1, 1)).view(feat4.size(0), -1)

        # Global Min Pooling (implemented as negative max of negative)
        min3 = -F.adaptive_max_pool2d(-feat3, (1, 1)).view(feat3.size(0), -1)
        min4 = -F.adaptive_max_pool2d(-feat4, (1, 1)).view(feat4.size(0), -1)

        # Concatenate all stats
        features = torch.cat([max3, min3, max4, min4], dim=1)
        return features

    def forward(self, x, inc_angle):
        """
        Full forward pass.
        Args:
            x: Image tensor (B, 3, 75, 75)
            inc_angle: Incidence angle tensor (B, 1) or (B,)
        """
        # Extract image features
        img_features = self.forward_features(x)

        # Ensure angle is correct shape (B, 1)
        if inc_angle.dim() == 1:
            inc_angle = inc_angle.view(-1, 1)

        # Fusion (Raw Scale)
        combined = torch.cat([img_features, inc_angle], dim=1)

        # Classification
        logits = self.classifier(combined)
        return logits
