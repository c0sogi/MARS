import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels.
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SelectiveSECNN(nn.Module):
    """
    Stabilized Selective-Hierarchical SE-CNN.

    Architecture:
    - 4-Stage Plain CNN Backbone (Conv-BN-ReLU-SE-MaxPool).
    - Early Channel Expansion (64 -> 128 -> 128 -> 128).
    - Selective Hierarchical Pooling: Global Max Pooling on Block 3 and Block 4 outputs.
    - Feature Fusion: Concatenates pooled features with raw incidence angle.
    - Single Hidden Layer Classification Head.
    """

    def __init__(self):
        super(SelectiveSECNN, self).__init__()

        # Hyperparameters from Config
        in_channels = Config.INPUT_CHANNELS
        widths = Config.CHANNEL_WIDTHS  # Expected: [64, 128, 128, 128]
        reduction = Config.SE_REDUCTION
        dropout_p = Config.FC_DROPOUT

        # ----------------------------------------------------------------------
        # Backbone
        # ----------------------------------------------------------------------

        # Block 1: Input -> 64
        self.conv1 = nn.Conv2d(
            in_channels, widths[0], kernel_size=3, padding=1, bias=True
        )
        self.bn1 = nn.BatchNorm2d(widths[0])
        self.se1 = SEBlock(widths[0], reduction)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: 64 -> 128 (Early Expansion)
        self.conv2 = nn.Conv2d(
            widths[0], widths[1], kernel_size=3, padding=1, bias=True
        )
        self.bn2 = nn.BatchNorm2d(widths[1])
        self.se2 = SEBlock(widths[1], reduction)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: 128 -> 128
        self.conv3 = nn.Conv2d(
            widths[1], widths[2], kernel_size=3, padding=1, bias=True
        )
        self.bn3 = nn.BatchNorm2d(widths[2])
        self.se3 = SEBlock(widths[2], reduction)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4: 128 -> 128
        self.conv4 = nn.Conv2d(
            widths[2], widths[3], kernel_size=3, padding=1, bias=True
        )
        self.bn4 = nn.BatchNorm2d(widths[3])
        self.se4 = SEBlock(widths[3], reduction)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ----------------------------------------------------------------------
        # Classification Head
        # ----------------------------------------------------------------------

        # Fusion Dimension:
        #   Block 3 Global Max Pool (128) +
        #   Block 4 Global Max Pool (128) +
        #   Incidence Angle (1)
        fusion_dim = widths[2] + widths[3] + 1
        hidden_dim = 256  # Standard hidden size for this capacity

        self.head_fc = nn.Linear(fusion_dim, hidden_dim)
        self.head_dropout = nn.Dropout(p=dropout_p)
        self.head_out = nn.Linear(hidden_dim, 1)

        # Initialization: Using PyTorch default (implicit Kaiming Uniform)

    def forward(self, x, angle):
        """
        Args:
            x: Image tensor (B, C, H, W)
            angle: Incidence angle tensor (B,)
        """
        # --- Stage 1 ---
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.se1(x)
        x = self.pool1(x)

        # --- Stage 2 ---
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.se2(x)
        x = self.pool2(x)

        # --- Stage 3 ---
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.se3(x)
        feat3 = self.pool3(x)  # Output of Block 3

        # --- Stage 4 ---
        x = self.conv4(feat3)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.se4(x)
        feat4 = self.pool4(x)  # Output of Block 4

        # --- Selective Hierarchical Pooling ---
        # Global Max Pooling to capture high-intensity signal peaks
        gmp3 = F.adaptive_max_pool2d(feat3, (1, 1)).view(feat3.size(0), -1)
        gmp4 = F.adaptive_max_pool2d(feat4, (1, 1)).view(feat4.size(0), -1)

        # --- Fusion ---
        # Reshape angle to (B, 1)
        angle = angle.view(-1, 1)

        # Concatenate features and raw angle
        fused = torch.cat([gmp3, gmp4, angle], dim=1)

        # --- Classifier ---
        out = self.head_fc(fused)
        out = F.relu(out)
        out = self.head_dropout(out)
        logits = self.head_out(out)

        return logits.squeeze(1)
