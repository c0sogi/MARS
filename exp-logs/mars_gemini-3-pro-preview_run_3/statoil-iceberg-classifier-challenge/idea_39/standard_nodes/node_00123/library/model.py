import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    IMAGE_SIZE,
    NUM_CHANNELS,
    LEAKY_RELU_SLOPE,
    DROPBLOCK_BLOCK_SIZE,
    DROPBLOCK_MAX_PROB,
    DROPBLOCK_START_PROB,
    DROPOUT_RATE,
)


class DropBlock2D(nn.Module):
    """
    DropBlock regularization to drop contiguous regions of features.
    """

    def __init__(self, block_size=DROPBLOCK_BLOCK_SIZE, drop_prob=DROPBLOCK_START_PROB):
        super(DropBlock2D, self).__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob

    def set_drop_prob(self, prob):
        self.drop_prob = prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x

        gamma = self._compute_gamma(x)
        mask = self._sample_mask(x, gamma)

        # Compute block mask by max pooling the sampled mask
        block_mask = self._compute_block_mask(mask)

        # Apply mask
        out = x * block_mask[:, None, :, :]

        # Normalize the features to maintain mean
        count = block_mask.numel()
        count_kept = block_mask.sum()
        return out * count / (count_kept + 1e-7)

    def _compute_gamma(self, x):
        _, _, H, W = x.shape
        feat_area = H * W
        block_area = self.block_size**2

        # valid region for sampling center of block
        valid_area = (H - self.block_size + 1) * (W - self.block_size + 1)

        return (self.drop_prob * feat_area) / (block_area * valid_area)

    def _sample_mask(self, x, gamma):
        batch_size, _, H, W = x.shape
        # Sample mask for the valid region where a block center can be placed
        p = (
            torch.ones(
                batch_size,
                H - self.block_size + 1,
                W - self.block_size + 1,
                device=x.device,
            )
            * gamma
        )
        return torch.bernoulli(p)

    def _compute_block_mask(self, mask):
        # mask is (N, H', W') where 1 indicates center of block to drop
        # We need to expand this to (N, H, W) where 0 indicates dropped pixel

        padding = self.block_size // 2

        # Pad the mask to match input dimensions after pooling
        mask_padded = F.pad(mask, (padding, padding, padding, padding), value=0)

        # Use max pool to dilate the drop centers into blocks
        # 1s in mask mean drop center. MaxPool spreads the 1s.
        mask_expanded = F.max_pool2d(
            mask_padded.unsqueeze(1), kernel_size=self.block_size, stride=1, padding=0
        )

        # Invert: 1 means keep, 0 means drop
        # mask_expanded has 1 where we should drop.
        return 1 - mask_expanded.squeeze(1)


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation Module with Global Average Pooling.
    """

    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
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


class DPDB_HSE_CNN(nn.Module):
    """
    Dual-Polarity DropBlock Hybrid-SE CNN.

    Architecture:
    - 4-Stage Plain CNN
    - LeakyReLU activations (preserve shadows)
    - SE Modules (Global Avg Pool)
    - DropBlock in Stages 3 & 4
    - Dual-Polarity Readout (Max & Min Pooling) from Stages 3 & 4
    - Fusion with raw incidence angle
    """

    def __init__(self):
        super(DPDB_HSE_CNN, self).__init__()

        # --- Stage 1 ---
        # 3 -> 64
        self.stage1 = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(LEAKY_RELU_SLOPE, inplace=True),
            SEModule(64),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 75 -> 37
        )

        # --- Stage 2 ---
        # 64 -> 128
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(LEAKY_RELU_SLOPE, inplace=True),
            SEModule(128),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 37 -> 18
        )

        # --- Stage 3 ---
        # 128 -> 128
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(LEAKY_RELU_SLOPE, inplace=True),
            SEModule(128),
        )
        self.dropblock3 = DropBlock2D(
            block_size=DROPBLOCK_BLOCK_SIZE, drop_prob=DROPBLOCK_START_PROB
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 18 -> 9

        # --- Stage 4 ---
        # 128 -> 128
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(LEAKY_RELU_SLOPE, inplace=True),
            SEModule(128),
        )
        self.dropblock4 = DropBlock2D(
            block_size=DROPBLOCK_BLOCK_SIZE, drop_prob=DROPBLOCK_START_PROB
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 9 -> 4

        # --- Classification Head ---
        # Input features calculation:
        # Stage 3 (128 ch): Max(128) + Min(128) = 256
        # Stage 4 (128 ch): Max(128) + Min(128) = 256
        # Angle: 1
        # Total: 513

        self.head = nn.Sequential(
            nn.Linear(513, 512),
            nn.LeakyReLU(LEAKY_RELU_SLOPE, inplace=True),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(512, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
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
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_dropblock_prob(self, progress):
        """
        Update DropBlock probability based on training progress (0.0 to 1.0).
        Linear schedule from START_PROB to MAX_PROB.
        """
        current_prob = DROPBLOCK_START_PROB + progress * (
            DROPBLOCK_MAX_PROB - DROPBLOCK_START_PROB
        )
        self.dropblock3.set_drop_prob(current_prob)
        self.dropblock4.set_drop_prob(current_prob)

    def forward(self, x, angle):
        # Stage 1
        x1 = self.stage1(x)

        # Stage 2
        x2 = self.stage2(x1)

        # Stage 3
        x3_feat = self.conv3(x2)
        x3_drop = self.dropblock3(x3_feat)
        x3_out = self.pool3(x3_drop)

        # Stage 4
        x4_feat = self.conv4(x3_out)
        x4_drop = self.dropblock4(x4_feat)
        x4_out = self.pool4(x4_drop)

        # --- Dual-Polarity Readout ---

        # Stage 3 Features (Global Max & Min)
        s3_max = F.adaptive_max_pool2d(x3_out, 1).view(x3_out.size(0), -1)
        # Min pooling implemented as Max(-x) to capture negative shadow values
        s3_min = F.adaptive_max_pool2d(-x3_out, 1).view(x3_out.size(0), -1)

        # Stage 4 Features (Global Max & Min)
        s4_max = F.adaptive_max_pool2d(x4_out, 1).view(x4_out.size(0), -1)
        s4_min = F.adaptive_max_pool2d(-x4_out, 1).view(x4_out.size(0), -1)

        # Concatenate Visual Features
        features = torch.cat([s3_max, s3_min, s4_max, s4_min], dim=1)

        # Concatenate Angle
        angle = angle.view(-1, 1)
        features = torch.cat([features, angle], dim=1)

        # Classification
        out = self.head(features)

        return out
