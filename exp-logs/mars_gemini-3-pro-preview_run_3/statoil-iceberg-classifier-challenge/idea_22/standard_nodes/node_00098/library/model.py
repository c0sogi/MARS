import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class MaxSEModule(nn.Module):
    """
    Max-Squeeze-and-Excitation Module.
    Uses Global Max Pooling to capture peak signals (icebergs) in the squeeze phase,
    rather than Average Pooling which might dilute sparse high-intensity features.
    """

    def __init__(self, channels, reduction=16):
        super(MaxSEModule, self).__init__()
        # Ensure hidden dimension is at least 1
        reduced_channels = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced_channels)
        self.fc2 = nn.Linear(reduced_channels, channels)

    def forward(self, x):
        batch, channels, _, _ = x.size()

        # Squeeze: Global Max Pooling
        # (N, C, H, W) -> (N, C)
        # This preserves the strongest signal in each channel
        y = F.adaptive_max_pool2d(x, 1).view(batch, channels)

        # Excitation: FC -> ReLU -> FC -> Sigmoid
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))

        # Scale: (N, C, 1, 1)
        y = y.view(batch, channels, 1, 1)

        return x * y


class MASHCNN(nn.Module):
    """
    Max-Attention Selective Hierarchical CNN (MASH-CNN).
    A 4-stage plain CNN optimized for peak signal detection in SAR imagery.
    """

    def __init__(self):
        super(MASHCNN, self).__init__()

        # ---------------------------
        # Feature Extractor (Backbone)
        # ---------------------------
        # Block 1: 3 -> 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.se1 = MaxSEModule(64)

        # Block 2: 64 -> 128 (Early Expansion to capture texture)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(128)
        self.se2 = MaxSEModule(128)

        # Block 3: 128 -> 128
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(128)
        self.se3 = MaxSEModule(128)

        # Block 4: 128 -> 128
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(128)
        self.se4 = MaxSEModule(128)

        # Shared Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---------------------------
        # Classification Head
        # ---------------------------
        # Fusion Input Dim:
        #   Stage 3 (128 channels) + Stage 4 (128 channels) + Angle (1 scalar)
        #   Total = 257
        self.fusion_dim = 128 + 128 + 1
        self.hidden_dim = 256

        self.fc1 = nn.Linear(self.fusion_dim, self.hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(self.hidden_dim, 1)

        # Initialization: PyTorch default (Kaiming Uniform) is used automatically.

    def forward(self, x, angle):
        # ---------------------------
        # Stage 1
        # ---------------------------
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.se1(x)
        x = self.pool(x)  # Output size: ~37x37

        # ---------------------------
        # Stage 2
        # ---------------------------
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.se2(x)
        x = self.pool(x)  # Output size: ~18x18

        # ---------------------------
        # Stage 3
        # ---------------------------
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.se3(x)
        x_s3 = self.pool(x)  # Output size: ~9x9. Save for fusion.

        # ---------------------------
        # Stage 4
        # ---------------------------
        x = self.conv4(x_s3)
        x = self.bn4(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.se4(x)
        x_s4 = self.pool(x)  # Output size: ~4x4. Save for fusion.

        # ---------------------------
        # Selective Hierarchical Pooling
        # ---------------------------
        # Global Max Pooling on Stage 3 and Stage 4 outputs
        feat_s3 = F.adaptive_max_pool2d(x_s3, 1).view(x.size(0), -1)  # (N, 128)
        feat_s4 = F.adaptive_max_pool2d(x_s4, 1).view(x.size(0), -1)  # (N, 128)

        # ---------------------------
        # Feature Fusion
        # ---------------------------
        # Ensure angle is (N, 1)
        angle = angle.view(-1, 1)

        # Concatenate: [Stage3, Stage4, Angle]
        fused = torch.cat([feat_s3, feat_s4, angle], dim=1)

        # ---------------------------
        # Classification Head
        # ---------------------------
        out = self.fc1(fused)
        out = F.leaky_relu(out, negative_slope=0.1)
        out = self.dropout(out)
        out = self.fc2(out)

        return out
