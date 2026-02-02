import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class MaxSEModule(nn.Module):
    """
    Max-Squeeze-and-Excitation Module.
    Uses Global Max Pooling to generate channel descriptors, ensuring high-intensity
    signals (like icebergs) are preserved and upweighted.
    """

    def __init__(self, channels, reduction=16):
        super(MaxSEModule, self).__init__()
        # Ensure hidden dimension is at least 1
        reduced_channels = max(1, channels // reduction)

        self.fc1 = nn.Linear(channels, reduced_channels, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduced_channels, channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global Max Pooling: (B, C, H, W) -> (B, C, 1, 1)
        # View as (B, C) for Linear layers
        y = F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)

        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)

        # Reshape to (B, C, 1, 1) for broadcasting
        y = y.view(x.size(0), x.size(1), 1, 1)

        # Scale input features
        return x * y


class DualBranchBlock(nn.Module):
    """
    Custom Dual-Branch Convolutional Block.
    Integrates Multi-Scale Dilated Convolutions, LeakyReLU, Max-SE Attention,
    and Spatial Downsampling.
    """

    def __init__(self, in_channels, out_channels):
        super(DualBranchBlock, self).__init__()

        # Local Branch: Captures fine-grained speckle/edge details
        # 3x3 Conv, dilation=1 -> padding=1 to maintain size before pooling
        self.local_branch = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=True
        )

        # Context Branch: Captures wider context (e.g., shadows)
        # 3x3 Conv, dilation=2 -> padding=2 to maintain size before pooling
        self.context_branch = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=True
        )

        # Fusion: Restores channel width after concatenation
        self.fusion = nn.Conv2d(
            out_channels * 2, out_channels, kernel_size=1, bias=True
        )

        # Activation: LeakyReLU preserves negative values (dB scale semantics)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # Attention: Max-Squeeze-and-Excitation
        self.se = MaxSEModule(out_channels)

        # Downsampling: Aggressive spatial reduction
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # 1. Multi-Scale Feature Extraction
        x_local = self.local_branch(x)
        x_context = self.context_branch(x)

        # 2. Fusion
        x_cat = torch.cat([x_local, x_context], dim=1)
        x_fused = self.fusion(x_cat)

        # 3. Activation
        x_act = self.act(x_fused)

        # 4. Attention
        x_se = self.se(x_act)

        # 5. Downsampling
        out = self.pool(x_se)

        return out


class MSMANet(nn.Module):
    """
    Multi-Scale Max-Attention Network (MSMA-Net).
    A 4-stage CNN optimized for iceberg detection using dual-branch blocks,
    selective hierarchical pooling, and raw angle fusion.
    """

    def __init__(self):
        super(MSMANet, self).__init__()

        # Input channels (HH, HV, Avg)
        in_channels = Config.IN_CHANNELS

        # Stage 1: 3 -> 64
        self.stage1 = DualBranchBlock(in_channels, 64)

        # Stage 2: 64 -> 128
        self.stage2 = DualBranchBlock(64, 128)

        # Stage 3: 128 -> 128
        self.stage3 = DualBranchBlock(128, 128)

        # Stage 4: 128 -> 128
        self.stage4 = DualBranchBlock(128, 128)

        # Classification Head
        # Inputs:
        #   - Stage 3 Global Max Pool (128)
        #   - Stage 4 Global Max Pool (128)
        #   - Incidence Angle (1)
        # Total Input Dim = 257
        head_input_dim = 128 + 128 + 1
        hidden_dim = 256  # Hidden layer dimension

        self.head = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(p=Config.DROPOUT_RATE),
            nn.Linear(hidden_dim, Config.NUM_CLASSES),
        )

    def forward(self, x, inc_angle):
        """
        Args:
            x (torch.Tensor): Input images (B, 3, 75, 75)
            inc_angle (torch.Tensor): Incidence angles (B,) or (B, 1)
        """
        # Backbone Forward Pass
        s1 = self.stage1(x)  # -> (B, 64, 37, 37)
        s2 = self.stage2(s1)  # -> (B, 128, 18, 18)
        s3 = self.stage3(s2)  # -> (B, 128, 9, 9)
        s4 = self.stage4(s3)  # -> (B, 128, 4, 4)

        # Selective Hierarchical Pooling
        # Global Max Pooling on Stage 3 Output
        p3 = F.adaptive_max_pool2d(s3, 1).view(s3.size(0), -1)  # (B, 128)

        # Global Max Pooling on Stage 4 Output
        p4 = F.adaptive_max_pool2d(s4, 1).view(s4.size(0), -1)  # (B, 128)

        # Prepare Incidence Angle
        if inc_angle.dim() == 1:
            angle = inc_angle.view(-1, 1)
        else:
            angle = inc_angle

        # Feature Fusion (Raw Scale)
        features = torch.cat([p3, p4, angle], dim=1)  # (B, 257)

        # Classification
        logits = self.head(features)

        return logits
