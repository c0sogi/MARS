import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Applies Channel Attention followed by Spatial Attention.
    """

    def __init__(self, channels, reduction=16):
        super(CBAMBlock, self).__init__()

        # Channel Attention: Shared MLP
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
        )

        # Spatial Attention: 7x7 Conv on concatenated Max+Avg maps
        self.conv7x7 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # --- Channel Attention ---
        b, c, h, w = x.size()
        # Compute Max and Avg pools for channel attention
        max_pool = F.max_pool2d(x, (h, w), stride=(h, w))
        avg_pool = F.avg_pool2d(x, (h, w), stride=(h, w))

        # Shared MLP processing
        channel_att = torch.sigmoid(self.mlp(max_pool) + self.mlp(avg_pool))
        channel_att = channel_att.view(b, c, 1, 1)

        # Apply channel attention
        x_out = x * channel_att

        # --- Spatial Attention ---
        # Compute Max and Avg pools along the channel dimension
        max_spatial, _ = torch.max(x_out, dim=1, keepdim=True)
        avg_spatial = torch.mean(x_out, dim=1, keepdim=True)

        # Concatenate and convolve
        spatial_input = torch.cat([max_spatial, avg_spatial], dim=1)
        spatial_att = torch.sigmoid(self.conv7x7(spatial_input))

        # Apply spatial attention
        return x_out * spatial_att


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


class DWB_DPN(nn.Module):
    """
    Dual-Pooling Network with Contracted Readout.
    Implements a wide backbone with Dual-Pooling and CBAM, followed by a
    contracted final layer and simple flattening to avoid overfitting.
    """

    def __init__(self):
        super(DWB_DPN, self).__init__()

        self.pool = DualPooling()

        # Stage 1: 3 -> 64. Output after DualPool: 128
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.cbam1 = CBAMBlock(64)

        # Stage 2: 128 -> 128. Output after DualPool: 256
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.cbam2 = CBAMBlock(128)

        # Stage 3: 256 -> 128. Output after DualPool: 256
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.cbam3 = CBAMBlock(128)

        # Stage 4: 256 -> 64 (Contraction). Output after DualPool: 128
        # Cite Lesson 00041: Prioritize Channel Contraction
        self.conv4 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.cbam4 = CBAMBlock(64)

        # Metadata Branch
        self.meta_fc = nn.Sequential(nn.Linear(1, 16), nn.BatchNorm1d(16), nn.ReLU())

        # Fusion Head
        # Visual: 128 * 4 * 4 = 2048
        # Meta: 16
        # Total: 2064
        self.fusion_fc = nn.Sequential(
            nn.Linear(2064, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(256, 1),
        )

    def forward(self, x, inc):
        # Stage 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.cbam1(x)
        x = self.pool(x)

        # Stage 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.cbam2(x)
        x = self.pool(x)

        # Stage 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.cbam3(x)
        x = self.pool(x)

        # Stage 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.cbam4(x)
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Metadata Branch
        inc = inc.view(-1, 1)
        xm = self.meta_fc(inc)

        # Fusion (Cite Lesson 00073: Simple Concatenation)
        fused = torch.cat([x, xm], dim=1)
        out = self.fusion_fc(fused)

        return out
