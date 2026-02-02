import torch
import torch.nn as nn
from library.config import IN_CHANNELS, STEM_CHANNELS, BRANCH_CHANNELS


class ResidualBlock(nn.Module):
    """
    Standard Residual Block with support for dilation.
    Structure: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> Add -> ReLU
    """

    def __init__(self, in_channels, out_channels, dilation):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection to handle channel changes
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class ParallelDilatedCNN(nn.Module):
    """
    Parallel-Scale Dilated CNN for Ink Detection.

    Features:
    1. Depth-to-Channel Stem with Instance Normalization.
    2. Three parallel branches with varying dilation rates (Texture, Stroke, Context).
    3. Fusion head to combine multi-scale features.
    """

    def __init__(self):
        super(ParallelDilatedCNN, self).__init__()

        # 1. Input Stem
        # Compresses 65 depth slices to STEM_CHANNELS (64)
        # Uses InstanceNorm to decouple from global intensity shifts
        self.stem = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, STEM_CHANNELS, kernel_size=1, bias=False),
            nn.InstanceNorm2d(STEM_CHANNELS, affine=True),
            nn.ReLU(inplace=True),
        )

        # 2. Parallel Branches
        # Each branch adapts the input (64 ch) to BRANCH_CHANNELS (16 ch) in the first block

        # Branch A: Texture Stream (Dilation = 1)
        # Focuses on high-frequency details
        self.branch_texture = nn.Sequential(
            ResidualBlock(STEM_CHANNELS, BRANCH_CHANNELS, dilation=1),
            ResidualBlock(BRANCH_CHANNELS, BRANCH_CHANNELS, dilation=1),
            ResidualBlock(BRANCH_CHANNELS, BRANCH_CHANNELS, dilation=1),
        )

        # Branch B: Stroke Stream (Dilation = 2, 4)
        # Focuses on stroke continuity
        self.branch_stroke = nn.Sequential(
            ResidualBlock(STEM_CHANNELS, BRANCH_CHANNELS, dilation=2),
            ResidualBlock(BRANCH_CHANNELS, BRANCH_CHANNELS, dilation=4),
            ResidualBlock(BRANCH_CHANNELS, BRANCH_CHANNELS, dilation=2),
        )

        # Branch C: Context Stream (Dilation = 4, 8)
        # Focuses on global shape and noise rejection
        self.branch_context = nn.Sequential(
            ResidualBlock(STEM_CHANNELS, BRANCH_CHANNELS, dilation=4),
            ResidualBlock(BRANCH_CHANNELS, BRANCH_CHANNELS, dilation=8),
            ResidualBlock(BRANCH_CHANNELS, BRANCH_CHANNELS, dilation=4),
        )

        # 3. Fusion & Classification
        # Concatenates outputs (16 * 3 = 48 channels)
        fusion_in_channels = BRANCH_CHANNELS * 3

        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in_channels, STEM_CHANNELS, kernel_size=1, bias=False),
            nn.BatchNorm2d(STEM_CHANNELS),
            nn.ReLU(inplace=True),
            nn.Conv2d(STEM_CHANNELS, 1, kernel_size=1),
            # Sigmoid is omitted here because BCEWithLogitsLoss is used in training.
            # Sigmoid is applied during inference.
        )

    def forward(self, x):
        # x shape: (Batch, 65, H, W)

        # Pass through stem
        features = self.stem(x)

        # Parallel processing
        tex = self.branch_texture(features)
        strk = self.branch_stroke(features)
        ctx = self.branch_context(features)

        # Concatenate along channel dimension
        concat = torch.cat([tex, strk, ctx], dim=1)

        # Fuse and predict
        logits = self.fusion(concat)

        return logits
