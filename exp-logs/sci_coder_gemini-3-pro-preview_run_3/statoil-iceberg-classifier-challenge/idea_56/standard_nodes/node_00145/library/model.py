import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class StatSE(nn.Module):
    """
    Statistical Squeeze-and-Excitation Module.
    Uses both Global Mean and Global Standard Deviation to recalibrate channels,
    capturing texture variance characteristic of SAR imagery.
    """

    def __init__(self, channels, reduction=16):
        super(StatSE, self).__init__()
        # Input to the MLP is 2 * channels (Mean + Std)
        # We ensure the bottleneck has at least 1 neuron
        bottleneck_dim = max(1, channels // reduction)

        self.fc1 = nn.Linear(channels * 2, bottleneck_dim, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(bottleneck_dim, channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        # Calculate Global Mean and Std
        # dim=(2, 3) reduces spatial dimensions (H, W)
        mean = x.mean(dim=(2, 3))  # Shape: (B, C)
        std = x.std(dim=(2, 3))  # Shape: (B, C)

        # Concatenate statistics
        stats = torch.cat([mean, std], dim=1)  # Shape: (B, 2*C)

        # Excitation (Attention generation)
        y = self.fc1(stats)
        y = self.act(y)
        y = self.fc2(y)
        y = self.sigmoid(y)  # Shape: (B, C)

        # Scale input
        return x * y.view(b, c, 1, 1)


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block for the Plain CNN backbone.
    Structure: Conv2d -> BN -> LeakyReLU -> StatSE -> MaxPool
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        # Bias is retained to preserve initialization dynamics
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=True
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(Config.LEAKY_RELU_SLOPE, inplace=True)
        self.se = StatSE(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        x = self.pool(x)
        return x


class DSICNN(nn.Module):
    """
    Decoupled Statistical-Isomorphic CNN (DSI-CNN).

    Key Features:
    1. Plain CNN Backbone (4 Stages).
    2. Statistical SE for texture awareness.
    3. Decoupled Readout: Separate projections for Stage 3 and Stage 4.
    4. Isomorphic Pooling: Max and Min pooling on the same projected features.
    5. Physics-informed fusion: Concatenates raw incidence angle.
    """

    def __init__(self):
        super(DSICNN, self).__init__()

        # --- Backbone ---
        # Stage 1
        self.block1 = ConvBlock(Config.IN_CHANNELS, Config.BLOCK_CHANNELS[0])
        # Stage 2
        self.block2 = ConvBlock(Config.BLOCK_CHANNELS[0], Config.BLOCK_CHANNELS[1])
        # Stage 3
        self.block3 = ConvBlock(Config.BLOCK_CHANNELS[1], Config.BLOCK_CHANNELS[2])
        # Stage 4
        self.block4 = ConvBlock(Config.BLOCK_CHANNELS[2], Config.BLOCK_CHANNELS[3])

        # --- Decoupled Isomorphic Readout ---
        # Separate 1x1 convolutions for Stage 3 and Stage 4
        # Reducing 128 channels -> 64 channels
        self.proj3 = nn.Conv2d(Config.BLOCK_CHANNELS[2], 64, kernel_size=1, bias=True)
        self.proj4 = nn.Conv2d(Config.BLOCK_CHANNELS[3], 64, kernel_size=1, bias=True)

        # --- Classification Head ---
        # Input Dimension:
        # Stage 3: 64 (Max) + 64 (Min) = 128
        # Stage 4: 64 (Max) + 64 (Min) = 128
        # Angle: 1
        # Total: 128 + 128 + 1 = 257 (Matches Config.FC_INPUT_DIM)

        self.fc1 = nn.Linear(Config.FC_INPUT_DIM, 256)
        self.act_head = nn.LeakyReLU(Config.LEAKY_RELU_SLOPE, inplace=True)
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)
        self.fc2 = nn.Linear(256, 1)  # Output logits

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        PyTorch Default Initialization (Kaiming Uniform / Fan-In).
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, angle):
        # --- Backbone Forward ---
        x = self.block1(x)
        x = self.block2(x)
        x3 = self.block3(x)  # Stage 3 Output
        x4 = self.block4(x3)  # Stage 4 Output

        # --- Decoupled Readout Stage 3 ---
        p3 = self.proj3(x3)
        # Global Max Pooling
        max3 = F.adaptive_max_pool2d(p3, 1).view(p3.size(0), -1)
        # Global Min Pooling (implemented as negative max of negative)
        min3 = -F.adaptive_max_pool2d(-p3, 1).view(p3.size(0), -1)

        # --- Decoupled Readout Stage 4 ---
        p4 = self.proj4(x4)
        # Global Max Pooling
        max4 = F.adaptive_max_pool2d(p4, 1).view(p4.size(0), -1)
        # Global Min Pooling
        min4 = -F.adaptive_max_pool2d(-p4, 1).view(p4.size(0), -1)

        # --- Fusion ---
        # Reshape angle to (B, 1)
        angle = angle.view(-1, 1)

        # Concatenate: [Max3, Min3, Max4, Min4, Angle]
        features = torch.cat([max3, min3, max4, min4, angle], dim=1)

        # --- Classification Head ---
        out = self.fc1(features)
        out = self.act_head(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out
