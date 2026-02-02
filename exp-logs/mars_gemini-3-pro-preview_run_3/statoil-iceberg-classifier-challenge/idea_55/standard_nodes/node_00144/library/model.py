import torch
import torch.nn as nn
import torch.nn.functional as F


class StatSE(nn.Module):
    """
    Statistical Squeeze-and-Excitation Module.
    Computes both Global Mean and Global Standard Deviation for channel recalibration.
    """

    def __init__(self, channels, reduction=16):
        super(StatSE, self).__init__()
        # Input to MLP is 2 * channels (Mean + Std)
        # We keep the bottleneck relative to the original channel count
        reduced_channels = max(channels // reduction, 4)

        self.fc1 = nn.Linear(channels * 2, reduced_channels, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduced_channels, channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, c, h, w = x.size()

        # Global Mean Pooling
        # (B, C, H, W) -> (B, C)
        feat_mean = x.view(batch, c, -1).mean(dim=2)

        # Global Std Pooling
        # (B, C, H, W) -> (B, C)
        feat_std = x.view(batch, c, -1).std(dim=2)

        # Concatenate statistics: (B, 2*C)
        stats = torch.cat([feat_mean, feat_std], dim=1)

        # Excitation
        out = self.fc1(stats)
        out = self.relu(out)
        out = self.fc2(out)
        scale = self.sigmoid(out)

        # Reshape for broadcasting: (B, C, 1, 1)
        scale = scale.view(batch, c, 1, 1)

        return x * scale


class ConvBlock(nn.Module):
    """
    Plain CNN Block: Conv -> BN -> LeakyReLU -> StatSE -> MaxPool
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        # Bias is retained as per description
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=True
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.se = StatSE(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        x = self.pool(x)
        return x


class SPCNN(nn.Module):
    """
    Statistical-Polarity Plain CNN (SP-CNN).
    Custom 4-Stage Attentive Convolutional Network with Isomorphic Dual-Polarity Readout.
    """

    def __init__(self):
        super(SPCNN, self).__init__()

        # Backbone Configuration
        # Input: 3 channels (HH, HV, Avg) -> 75x75

        # Stage 1: 3 -> 64
        self.stage1 = ConvBlock(3, 64)

        # Stage 2: 64 -> 128
        self.stage2 = ConvBlock(64, 128)

        # Stage 3: 128 -> 128
        self.stage3 = ConvBlock(128, 128)

        # Stage 4: 128 -> 128
        self.stage4 = ConvBlock(128, 128)

        # Isomorphic Dual-Polarity Readout
        # Shared projection 1x1 Conv: 128 -> 64
        self.shared_projector = nn.Conv2d(128, 64, kernel_size=1, bias=True)

        # Classification Head
        # Features:
        #   Stage 3: 64 (Max) + 64 (Min) = 128
        #   Stage 4: 64 (Max) + 64 (Min) = 128
        #   Total CNN Features: 256
        #   Incidence Angle: 1
        #   Total Input: 257
        self.classifier = nn.Sequential(
            nn.Linear(257, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

        self._init_weights()

    def _init_weights(self):
        # PyTorch Default Initialization (Kaiming Uniform)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, angle):
        """
        Args:
            x (torch.Tensor): Image input (B, 3, 75, 75)
            angle (torch.Tensor): Incidence angle input (B,) or (B, 1)
        """
        # Backbone
        x1 = self.stage1(x)  # 75 -> 37
        x2 = self.stage2(x1)  # 37 -> 18
        x3 = self.stage3(x2)  # 18 -> 9
        x4 = self.stage4(x3)  # 9 -> 4

        # Readout from Stage 3 and Stage 4
        # Apply shared projection
        f3 = self.shared_projector(x3)  # (B, 64, 9, 9)
        f4 = self.shared_projector(x4)  # (B, 64, 4, 4)

        # Dual-Polarity Pooling (Max and Min)
        # Flatten spatial dims for pooling
        b, c, _, _ = f3.size()
        f3_flat = f3.view(b, c, -1)
        f4_flat = f4.view(b, 64, -1)  # f4 has same channels (64)

        # Max Pooling
        v3_max = f3_flat.max(dim=2)[0]
        v4_max = f4_flat.max(dim=2)[0]

        # Min Pooling (implemented as -max(-x) or min(dim).values)
        v3_min = f3_flat.min(dim=2)[0]
        v4_min = f4_flat.min(dim=2)[0]

        # Concatenate features
        cnn_features = torch.cat([v3_max, v3_min, v4_max, v4_min], dim=1)  # (B, 256)

        # Fusion with Incidence Angle
        # Ensure angle is (B, 1)
        if angle.dim() == 1:
            angle = angle.view(-1, 1)

        # Concatenate raw angle (no normalization as per idea)
        final_features = torch.cat([cnn_features, angle], dim=1)  # (B, 257)

        # Classification
        logits = self.classifier(final_features)

        return logits.view(-1)
