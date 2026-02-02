import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DualPooling(nn.Module):
    """
    Dual-Stream Pooling.
    Concatenates Max Pooling (Peaks) and Min Pooling (Shadows).
    Doubles the channel dimension.
    """

    def __init__(self):
        super(DualPooling, self).__init__()
        self.max_pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        # Max Pooling captures signal peaks
        p_max = self.max_pool(x)
        # Min Pooling captures signal shadows (implemented as -MaxPool(-x))
        p_min = -self.max_pool(-x)
        return torch.cat([p_max, p_min], dim=1)


class WideBodyBackbone(nn.Module):
    """
    Wide-Body Backbone with 4 stages.
    Maintains high channel width but removes CBAM to avoid interference with DualPooling.
    Contracts in the final stage to manage parameter count.
    """

    def __init__(self):
        super(WideBodyBackbone, self).__init__()

        self.pool = DualPooling()

        # Stage 1: 3 -> 64. Output after DualPool: 128
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Stage 2: 128 -> 128. Output after DualPool: 256
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # Stage 3: 256 -> 128. Output after DualPool: 256
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Stage 4: 256 -> 64 (Contraction). Output after DualPool: 128
        self.conv4 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

    def forward(self, x):
        # Stage 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        # Stage 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Stage 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        # Stage 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        return x


class DWB_DPN(nn.Module):
    """
    Simplified Wide-Body Dual-Pooling Network.
    Uses standard flattening and fusion.
    """

    def __init__(self):
        super(DWB_DPN, self).__init__()

        self.backbone = WideBodyBackbone()

        # Metadata Branch
        self.meta_fc = nn.Sequential(nn.Linear(1, 16), nn.BatchNorm1d(16), nn.ReLU())

        # Fusion Head
        # Visual: 128 * 4 * 4 = 2048
        # Meta: 16
        # Total: 2064
        self.fusion_fc = nn.Sequential(
            nn.Linear(2064, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(256, 1),
        )

    def forward(self, x, inc):
        # Visual Branch
        features = self.backbone(x)
        # Flatten: (B, 128, 4, 4) -> (B, 2048)
        xv = features.view(features.size(0), -1)

        # Metadata Branch
        inc = inc.view(-1, 1)
        xm = self.meta_fc(inc)

        # Fusion
        fused = torch.cat([xv, xm], dim=1)
        out = self.fusion_fc(fused)

        return out
