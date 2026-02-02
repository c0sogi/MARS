import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridSE(nn.Module):
    """
    Hybrid Squeeze-and-Excitation Module.
    Uses Global Average Pooling for the squeeze operation to act as a low-pass filter.
    """

    def __init__(self, channels, reduction=16):
        super(HybridSE, self).__init__()
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


class IsomorphicReadout(nn.Module):
    """
    Isomorphic Dual-Polarity Readout Layer.
    Projects channels to a lower dimension, then applies both Global Max Pooling (Peaks)
    and Global Min Pooling (Shadows) to capture signal contrast while maintaining
    fixed output dimensionality.
    """

    def __init__(self, in_channels, out_channels_per_pool=64):
        super(IsomorphicReadout, self).__init__()
        # Project to half the desired output size because we concat Max and Min
        self.project = nn.Conv2d(
            in_channels, out_channels_per_pool, kernel_size=1, bias=True
        )

    def forward(self, x):
        # x: (B, C_in, H, W)
        x_proj = self.project(x)  # (B, 64, H, W)

        # Global Max Pooling (Peaks)
        max_pool = F.adaptive_max_pool2d(x_proj, (1, 1)).view(x.size(0), -1)

        # Global Min Pooling (Shadows) -> Max(-x)
        # We negate x, take the max (which corresponds to the magnitude of the min),
        # and then negate back to preserve the original sign/value of the minimum.
        min_pool = F.adaptive_max_pool2d(-x_proj, (1, 1)).view(x.size(0), -1)
        min_pool = -min_pool

        # Concatenate: 64 + 64 = 128 features
        return torch.cat([max_pool, min_pool], dim=1)


class IDPH_CNN(nn.Module):
    """
    Isomorphic Dual-Polarity Hierarchical CNN.
    A 4-stage Plain CNN with Hybrid SE blocks and a dual-polarity readout
    designed to capture iceberg peaks and radar shadows.
    """

    def __init__(self):
        super(IDPH_CNN, self).__init__()

        # Stage 1
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            HybridSE(64),
            nn.MaxPool2d(2, 2),
        )

        # Stage 2
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            HybridSE(128),
            nn.MaxPool2d(2, 2),
        )

        # Stage 3
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            HybridSE(128),
            nn.MaxPool2d(2, 2),
        )
        self.readout3 = IsomorphicReadout(128, 64)  # Output 128 features

        # Stage 4
        self.stage4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            HybridSE(128),
            nn.MaxPool2d(2, 2),
        )
        self.readout4 = IsomorphicReadout(128, 64)  # Output 128 features

        # Classifier
        # Input: 128 (Stage3) + 128 (Stage4) + 1 (Angle) = 257
        self.classifier = nn.Sequential(
            nn.Linear(257, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

        # Initialization (Kaiming Uniform / Fan-In)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, angle):
        # Backbone
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        # Readouts
        feat3 = self.readout3(x3)
        feat4 = self.readout4(x4)

        # Fusion
        # Ensure angle is (B, 1)
        angle = angle.view(-1, 1)
        features = torch.cat([feat3, feat4, angle], dim=1)

        # Classification
        out = self.classifier(features)
        return out
