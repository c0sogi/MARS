import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DropBlock2D(nn.Module):
    """
    DropBlock2D regularization layer.
    Drops contiguous spatial regions to force the network to learn distributed features.
    """

    def __init__(self, drop_prob=0.1, block_size=3):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # Only apply during training and if drop_prob > 0
        if not self.training or self.drop_prob == 0.0:
            return x

        N, C, H, W = x.shape

        # Calculate gamma
        # gamma = (drop_prob * total_area) / (block_area * valid_area)
        total_area = H * W
        block_area = self.block_size**2
        valid_h = H - self.block_size + 1
        valid_w = W - self.block_size + 1

        # If the feature map is smaller than the block size, skip dropping
        if valid_h <= 0 or valid_w <= 0:
            return x

        valid_area = valid_h * valid_w
        gamma = (self.drop_prob * total_area) / (block_area * valid_area)

        # Sample mask centers from Bernoulli distribution
        mask_centers = torch.bernoulli(
            torch.full((N, 1, valid_h, valid_w), gamma, device=x.device)
        )

        # Pad to match original size so that max_pool results in HxW
        padding = self.block_size // 2

        # Create block mask using MaxPool
        # We want to place a 1 at the center and expand it to block_size
        mask = F.max_pool2d(
            mask_centers, kernel_size=self.block_size, stride=1, padding=padding
        )

        # Ensure dimensions match (handle potential odd/even size mismatches)
        if mask.shape[2] != H or mask.shape[3] != W:
            mask = F.interpolate(mask, size=(H, W), mode="nearest")

        # Invert mask: 1 means drop, so we want to keep where mask is 0
        # mask becomes 1 for keep, 0 for drop
        mask = 1 - mask

        # Normalize features to preserve mean
        # Normalize by (total elements / elements kept)
        count_kept = mask.sum()
        if count_kept == 0:
            return x * 0

        normalize_scale = mask.numel() / count_kept

        return x * mask * normalize_scale

    def set_drop_prob(self, prob):
        """Sets the drop probability, useful for scheduling."""
        self.drop_prob = prob


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation Module.
    Uses Global Average Pooling for Squeeze and a 2-layer MLP for Excitation.
    """

    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=True),
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


class DualPolarityDropBlockSECNN(nn.Module):
    """
    Custom 4-Stage Attentive Convolutional Network.

    Architecture:
    - Backbone: Plain CNN (4 stages, no residuals)
    - Activation: LeakyReLU (negative slope 0.1)
    - Attention: SE Modules in all stages
    - Regularization: DropBlock2D in Stages 3 and 4
    - Readout: Dual-Polarity (Global Max + Global Min) from Stages 3 and 4
    - Fusion: Concatenation of features + raw incidence angle
    """

    def __init__(self):
        super(DualPolarityDropBlockSECNN, self).__init__()

        self.channels = Config.LAYER_CHANNELS  # [64, 128, 128, 128]
        self.slope = Config.LEAKY_RELU_SLOPE

        # --- Stage 1 ---
        # Input: (B, 3, 75, 75) -> Output: (B, 64, 37, 37)
        self.stage1_conv = nn.Conv2d(
            Config.IN_CHANNELS, self.channels[0], kernel_size=3, padding=1, bias=True
        )
        self.stage1_bn = nn.BatchNorm2d(self.channels[0])
        self.stage1_act = nn.LeakyReLU(self.slope, inplace=True)
        self.stage1_se = SEModule(self.channels[0])
        self.stage1_pool = nn.MaxPool2d(2, 2)

        # --- Stage 2 ---
        # Input: (B, 64, 37, 37) -> Output: (B, 128, 18, 18)
        self.stage2_conv = nn.Conv2d(
            self.channels[0], self.channels[1], kernel_size=3, padding=1, bias=True
        )
        self.stage2_bn = nn.BatchNorm2d(self.channels[1])
        self.stage2_act = nn.LeakyReLU(self.slope, inplace=True)
        self.stage2_se = SEModule(self.channels[1])
        self.stage2_pool = nn.MaxPool2d(2, 2)

        # --- Stage 3 ---
        # Input: (B, 128, 18, 18) -> Output: (B, 128, 9, 9)
        self.stage3_conv = nn.Conv2d(
            self.channels[1], self.channels[2], kernel_size=3, padding=1, bias=True
        )
        self.stage3_bn = nn.BatchNorm2d(self.channels[2])
        self.stage3_act = nn.LeakyReLU(self.slope, inplace=True)
        self.stage3_se = SEModule(self.channels[2])
        self.stage3_drop = DropBlock2D(
            drop_prob=0.0, block_size=Config.DROPBLOCK_BLOCK_SIZE
        )
        self.stage3_pool = nn.MaxPool2d(2, 2)

        # --- Stage 4 ---
        # Input: (B, 128, 9, 9) -> Output: (B, 128, 4, 4)
        self.stage4_conv = nn.Conv2d(
            self.channels[2], self.channels[3], kernel_size=3, padding=1, bias=True
        )
        self.stage4_bn = nn.BatchNorm2d(self.channels[3])
        self.stage4_act = nn.LeakyReLU(self.slope, inplace=True)
        self.stage4_se = SEModule(self.channels[3])
        self.stage4_drop = DropBlock2D(
            drop_prob=0.0, block_size=Config.DROPBLOCK_BLOCK_SIZE
        )
        self.stage4_pool = nn.MaxPool2d(2, 2)

        # --- Classification Head ---
        # Features:
        # Stage 3 (128 ch): Max + Min = 256
        # Stage 4 (128 ch): Max + Min = 256
        # Angle: 1
        # Total Input: 513
        self.head_input_dim = (self.channels[2] * 2) + (self.channels[3] * 2) + 1
        self.head_hidden_dim = 256

        self.classifier = nn.Sequential(
            nn.Linear(self.head_input_dim, self.head_hidden_dim),
            nn.LeakyReLU(self.slope, inplace=True),
            nn.Dropout(Config.CLASSIFIER_DROPOUT),
            nn.Linear(self.head_hidden_dim, 1),
        )

    def forward(self, x, angle):
        # x: (B, 3, 75, 75)
        # angle: (B,)

        # --- Stage 1 ---
        x = self.stage1_conv(x)
        x = self.stage1_bn(x)
        x = self.stage1_act(x)
        x = self.stage1_se(x)
        x = self.stage1_pool(x)

        # --- Stage 2 ---
        x = self.stage2_conv(x)
        x = self.stage2_bn(x)
        x = self.stage2_act(x)
        x = self.stage2_se(x)
        x = self.stage2_pool(x)

        # --- Stage 3 ---
        x = self.stage3_conv(x)
        x = self.stage3_bn(x)
        x = self.stage3_act(x)
        x = self.stage3_se(x)
        x = self.stage3_drop(x)

        # Extract Stage 3 Features (before pooling to retain spatial resolution for global pooling)
        x3_feat = x

        # Apply structural pool for next stage
        x = self.stage3_pool(x)

        # --- Stage 4 ---
        x = self.stage4_conv(x)
        x = self.stage4_bn(x)
        x = self.stage4_act(x)
        x = self.stage4_se(x)
        x = self.stage4_drop(x)

        # Extract Stage 4 Features
        x4_feat = x

        # (Structural pool for Stage 4 is not strictly needed for readout, but defined in arch)
        # x = self.stage4_pool(x)

        # --- Dual-Polarity Pooling ---
        # We use adaptive_max_pool2d(x, 1) for Global Max
        # We use adaptive_max_pool2d(-x, 1) for Global Min (Shadow Depth)

        # Stage 3 Readout
        s3_max = F.adaptive_max_pool2d(x3_feat, 1).view(x3_feat.size(0), -1)
        s3_min = F.adaptive_max_pool2d(-x3_feat, 1).view(x3_feat.size(0), -1)

        # Stage 4 Readout
        s4_max = F.adaptive_max_pool2d(x4_feat, 1).view(x4_feat.size(0), -1)
        s4_min = F.adaptive_max_pool2d(-x4_feat, 1).view(x4_feat.size(0), -1)

        # --- Fusion ---
        angle = angle.view(-1, 1)
        features = torch.cat([s3_max, s3_min, s4_max, s4_min, angle], dim=1)

        # --- Classification ---
        out = self.classifier(features)

        return out

    def set_dropblock_prob(self, prob):
        """
        Updates the drop probability for DropBlock layers.
        Should be called by the training loop to implement the schedule.
        """
        self.stage3_drop.set_drop_prob(prob)
        self.stage4_drop.set_drop_prob(prob)
