import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block using Global Average Pooling.
    Recalibrates channel-wise feature responses.
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Scale
        return x * y


class DropBlock2D(nn.Module):
    """
    DropBlock regularization module.
    Drops contiguous regions (blocks) of the feature map to prevent overfitting
    to specific local features (speckle noise).
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # Only apply during training and if probability > 0
        if not self.training or self.drop_prob == 0.0:
            return x

        N, C, H, W = x.size()

        # Calculate gamma (Bernoulli probability for mask centers)
        # gamma = (drop_prob * H * W) / (block_size^2 * (H - block_size + 1) * (W - block_size + 1))
        total_size = H * W
        clipped_block_size = min(self.block_size, min(H, W))

        valid_h = H - clipped_block_size + 1
        valid_w = W - clipped_block_size + 1

        if valid_h <= 0 or valid_w <= 0:
            return x

        gamma = (self.drop_prob * total_size) / (
            (clipped_block_size**2) * valid_h * valid_w
        )

        # Sample mask centers on the valid region
        mask = torch.bernoulli(
            torch.full((N, C, valid_h, valid_w), gamma, device=x.device)
        )

        # Pad the mask so that after max_pool expansion it matches input size
        pad_h = (clipped_block_size - 1) // 2
        pad_w = (clipped_block_size - 1) // 2

        mask_padded = F.pad(mask, (pad_w, pad_w, pad_h, pad_h), value=0)

        # Expand the mask using MaxPool (dilation of the sampled points)
        block_mask = F.max_pool2d(
            mask_padded,
            kernel_size=clipped_block_size,
            stride=1,
            padding=clipped_block_size // 2,
        )

        # Ensure dimensions match (handle potential rounding issues)
        if block_mask.shape[2] != H or block_mask.shape[3] != W:
            block_mask = block_mask[:, :, :H, :W]

        # Invert mask: 1 means drop, so we keep where mask is 0
        keep_mask = 1 - block_mask

        # Normalize the features to maintain the mean activation magnitude
        scale = keep_mask.numel() / (keep_mask.sum() + 1e-7)

        return x * keep_mask * scale


class ProjectedDualReadout(nn.Module):
    """
    Projects channels to a lower dimension, then applies parallel
    Global Max Pooling (Peak) and Global Min Pooling (Shadow).
    """

    def __init__(self, in_channels, out_channels):
        super(ProjectedDualReadout, self).__init__()
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        # Project: (B, C_in, H, W) -> (B, C_out, H, W)
        x = self.project(x)

        # Global Max Pooling: (B, C_out)
        max_pool = F.adaptive_max_pool2d(x, (1, 1)).view(x.size(0), -1)

        # Global Min Pooling: (B, C_out)
        # Implemented as -Max(-x)
        min_pool = -F.adaptive_max_pool2d(-x, (1, 1)).view(x.size(0), -1)

        # Concatenate: (B, 2 * C_out)
        return torch.cat([max_pool, min_pool], dim=1)


class IcebergModel(nn.Module):
    """
    Projected Dual-Polarity DropBlock CNN (PDP-D-CNN).
    A 4-stage plain CNN with SE blocks, DropBlock regularization,
    and multi-scale dual-polarity readout.
    """

    def __init__(self):
        super(IcebergModel, self).__init__()

        self.filters = Config.BACKBONE_FILTERS  # [64, 128, 128, 128]

        # --- Stage 1 ---
        self.stage1 = nn.Sequential(
            nn.Conv2d(
                Config.IN_CHANNELS,
                self.filters[0],
                kernel_size=3,
                padding=1,
                bias=Config.USE_BIAS,
            ),
            nn.BatchNorm2d(self.filters[0]),
            nn.LeakyReLU(Config.LEAKY_RELU_SLOPE, inplace=True),
            SEBlock(self.filters[0]),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # --- Stage 2 ---
        self.stage2 = nn.Sequential(
            nn.Conv2d(
                self.filters[0],
                self.filters[1],
                kernel_size=3,
                padding=1,
                bias=Config.USE_BIAS,
            ),
            nn.BatchNorm2d(self.filters[1]),
            nn.LeakyReLU(Config.LEAKY_RELU_SLOPE, inplace=True),
            SEBlock(self.filters[1]),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # --- Stage 3 (with DropBlock) ---
        self.stage3_conv = nn.Sequential(
            nn.Conv2d(
                self.filters[1],
                self.filters[2],
                kernel_size=3,
                padding=1,
                bias=Config.USE_BIAS,
            ),
            nn.BatchNorm2d(self.filters[2]),
            nn.LeakyReLU(Config.LEAKY_RELU_SLOPE, inplace=True),
            SEBlock(self.filters[2]),
        )
        self.stage3_drop = DropBlock2D(
            Config.DROPBLOCK_START_PROB, Config.DROPBLOCK_BLOCK_SIZE
        )
        self.stage3_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Stage 4 (with DropBlock) ---
        self.stage4_conv = nn.Sequential(
            nn.Conv2d(
                self.filters[2],
                self.filters[3],
                kernel_size=3,
                padding=1,
                bias=Config.USE_BIAS,
            ),
            nn.BatchNorm2d(self.filters[3]),
            nn.LeakyReLU(Config.LEAKY_RELU_SLOPE, inplace=True),
            SEBlock(self.filters[3]),
        )
        self.stage4_drop = DropBlock2D(
            Config.DROPBLOCK_START_PROB, Config.DROPBLOCK_BLOCK_SIZE
        )
        self.stage4_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Readouts ---
        # Applied before pooling at Stage 3 and Stage 4 to capture spatial details
        self.readout3 = ProjectedDualReadout(self.filters[2], Config.PROJECTION_DIM)
        self.readout4 = ProjectedDualReadout(self.filters[3], Config.PROJECTION_DIM)

        # --- Classification Head ---
        # Input Dimension Calculation:
        # Stage 3 Readout: 64 (Max) + 64 (Min) = 128
        # Stage 4 Readout: 64 (Max) + 64 (Min) = 128
        # Incidence Angle: 1
        # Total: 128 + 128 + 1 = 257
        input_dim = (Config.PROJECTION_DIM * 2) * 2 + 1

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(Config.LEAKY_RELU_SLOPE, inplace=True),
            nn.Dropout(Config.FC_DROPOUT),
            nn.Linear(256, 1),
        )

    def forward(self, x, angle):
        """
        Args:
            x (torch.Tensor): Image input of shape (B, 3, 75, 75)
            angle (torch.Tensor): Incidence angle of shape (B,)
        """
        # Stage 1
        x = self.stage1(x)

        # Stage 2
        x = self.stage2(x)

        # Stage 3
        x = self.stage3_conv(x)
        x = self.stage3_drop(x)
        feat3 = x  # Capture features for readout before pooling
        x = self.stage3_pool(x)

        # Stage 4
        x = self.stage4_conv(x)
        x = self.stage4_drop(x)
        feat4 = x  # Capture features for readout before pooling
        x = self.stage4_pool(x)

        # Readouts
        r3 = self.readout3(feat3)  # (B, 128)
        r4 = self.readout4(feat4)  # (B, 128)

        # Angle Processing
        ang = angle.view(-1, 1)  # (B, 1)

        # Feature Fusion
        out = torch.cat([r3, r4, ang], dim=1)  # (B, 257)

        # Classification
        logits = self.classifier(out)
        return logits.squeeze(1)  # (B,)

    def set_dropblock_prob(self, prob):
        """
        Updates the drop probability for DropBlock layers.
        Used by the training loop to implement linear scheduling.
        """
        self.stage3_drop.drop_prob = prob
        self.stage4_drop.drop_prob = prob
