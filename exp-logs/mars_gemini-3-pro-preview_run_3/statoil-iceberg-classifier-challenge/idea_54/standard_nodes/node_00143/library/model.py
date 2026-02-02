import torch
import torch.nn as nn
import torch.nn.functional as F


class StatSELayer(nn.Module):
    """
    Statistical Squeeze-and-Excitation Module.
    Fuses Global Average Pooling (Mean) and Global Standard Deviation Pooling (Texture)
    to recalibrate channel weights.
    """

    def __init__(self, channels, reduction=16):
        super(StatSELayer, self).__init__()
        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Bottleneck dimension
        bottleneck_dim = max(1, channels // reduction)

        # MLP: Input is 2 * channels (Mean + Std)
        # We explicitly retain bias terms as per architecture requirements
        self.fc = nn.Sequential(
            nn.Linear(2 * channels, bottleneck_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # 1. Statistical Squeeze
        # Mean: (B, C, 1, 1) -> (B, C)
        y_mean = self.avg_pool(x).view(b, c)

        # Std: (B, C)
        # Compute standard deviation over spatial dimensions (H, W)
        # unbiased=False (1/N) is used to be consistent with standard deep learning moments (e.g. BatchNorm)
        y_std = torch.std(x, dim=(2, 3), unbiased=False)

        # Concatenate Statistics: (B, 2*C)
        y = torch.cat([y_mean, y_std], dim=1)

        # 2. Excitation
        y = self.fc(y).view(b, c, 1, 1)

        # 3. Scale
        return x * y


class SAICNN(nn.Module):
    """
    Statistical-Attentive Isomorphic CNN (SAI-CNN).

    Architecture:
    - Backbone: 4-Stage Plain CNN with Stat-SE modules.
    - Readout: Isomorphic Dual-Polarity (Max + Min) on Shared Projection from Stages 3 & 4.
    - Head: Concatenates raw scalar incidence angle.
    """

    def __init__(self):
        super(SAICNN, self).__init__()

        # --- Backbone (4 Stages) ---
        # Strategy: Early Expansion (64 -> 128 -> 128 -> 128)
        # Downsampling: 75 -> 37 -> 18 -> 9 -> 4

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            StatSELayer(64, reduction=16),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            StatSELayer(128, reduction=16),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            StatSELayer(128, reduction=16),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.stage4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            StatSELayer(128, reduction=16),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # --- Readout ---
        # Shared 1x1 Convolution to compress channels (128 -> 64)
        # This layer is applied identically to outputs of Stage 3 and Stage 4
        self.shared_proj = nn.Conv2d(128, 64, kernel_size=1, bias=True)

        # --- Classification Head ---
        # Features:
        # Stage 3: 64(Max) + 64(Min) = 128
        # Stage 4: 64(Max) + 64(Min) = 128
        # Angle: 1
        # Total Input: 257
        self.classifier = nn.Sequential(
            nn.Linear(257, 256, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1, bias=True),
        )

    def forward(self, x, angle):
        """
        Args:
            x (torch.Tensor): Image input of shape (B, 3, 75, 75).
            angle (torch.Tensor): Incidence angle of shape (B,).
        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        # Backbone Forward Pass
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)  # Output of Stage 3 (128, 9, 9)
        x4 = self.stage4(x3)  # Output of Stage 4 (128, 4, 4)

        # Readout: Shared Projection
        p3 = self.shared_proj(x3)  # (B, 64, 9, 9)
        p4 = self.shared_proj(x4)  # (B, 64, 4, 4)

        # Readout: Dual-Polarity Pooling
        # Global Max Pooling (Peaks)
        max_3 = F.adaptive_max_pool2d(p3, (1, 1)).view(p3.size(0), -1)
        max_4 = F.adaptive_max_pool2d(p4, (1, 1)).view(p4.size(0), -1)

        # Global Min Pooling (Shadows)
        # Implemented as Max(-x) to capture the magnitude of negative peaks as positive features
        min_3 = F.adaptive_max_pool2d(-p3, (1, 1)).view(p3.size(0), -1)
        min_4 = F.adaptive_max_pool2d(-p4, (1, 1)).view(p4.size(0), -1)

        # Concatenate Image Features
        img_feats = torch.cat([max_3, min_3, max_4, min_4], dim=1)  # Size: 256

        # Process Angle
        angle = angle.view(-1, 1)  # Size: (B, 1)

        # Feature Fusion
        features = torch.cat([img_feats, angle], dim=1)  # Size: 257

        # Classification
        out = self.classifier(features)

        return out
