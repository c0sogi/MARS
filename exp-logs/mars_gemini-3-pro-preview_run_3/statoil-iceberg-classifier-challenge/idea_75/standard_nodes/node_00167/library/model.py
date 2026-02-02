import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class HybridSE(nn.Module):
    """
    Hybrid Squeeze-and-Excitation Module.
    Uses Global Average Pooling for the squeeze operation (low-pass filter).
    """

    def __init__(self, channels, reduction=16):
        super(HybridSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block for CCTI-CNN.
    Structure: Conv2d(bias=True) -> BN -> LeakyReLU -> SE -> MaxPool
    """

    def __init__(self, in_channels, out_channels, reduction=16):
        super(ConvBlock, self).__init__()
        # Explicitly retaining bias=True as per Lesson 76
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=True
        )
        self.bn = nn.BatchNorm2d(out_channels)
        # LeakyReLU with negative slope 0.1 as per Lesson 91
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.se = HybridSE(out_channels, reduction=reduction)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        x = self.pool(x)
        return x


class CCTICNN(nn.Module):
    """
    Capacity-Constrained Tri-Statistic Isomorphic CNN.
    """

    def __init__(self):
        super(CCTICNN, self).__init__()

        # --- Backbone ---
        # 4-Stage Plain CNN
        self.stages = nn.ModuleList()
        in_c = Config.IN_CHANNELS

        # Build stages based on config channels [64, 128, 128, 128]
        for out_c in Config.BACKBONE_CHANNELS:
            self.stages.append(ConvBlock(in_c, out_c, reduction=Config.SE_REDUCTION))
            in_c = out_c

        # --- Isomorphic Readout Projections ---
        # We extract from indices defined in Config (usually [2, 3])
        # We need separate projections for each extracted stage to learn depth-specific transforms.
        # Input dim is 128 (from backbone), Output is 64 (PROJECTION_DIM).
        self.extract_indices = Config.EXTRACT_INDICES
        self.projections = nn.ModuleList()

        # Assuming the channels at extracted stages match the backbone config
        for idx in self.extract_indices:
            stage_channels = Config.BACKBONE_CHANNELS[idx]
            proj = nn.Conv2d(
                stage_channels, Config.PROJECTION_DIM, kernel_size=1, bias=True
            )
            self.projections.append(proj)

        # --- Capacity-Constrained Head ---
        # Feature calculation:
        # Num extracted stages (2) * Num stats (3) * Projection Dim (64) = 384
        # Plus 1 for incidence angle = 385
        num_stats = 3  # Max, Min, MAD
        self.feature_dim = len(self.extract_indices) * num_stats * Config.PROJECTION_DIM
        input_dim = self.feature_dim + 1  # +1 for angle

        self.head = nn.Sequential(
            nn.Linear(input_dim, Config.HEAD_HIDDEN_DIM),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(p=Config.DROPOUT_RATE),
            nn.Linear(Config.HEAD_HIDDEN_DIM, Config.NUM_CLASSES),
        )

        # Initialization is handled by PyTorch defaults (Kaiming Uniform)

    def _tri_stat_pooling(self, x):
        """
        Applies Tri-Statistic Pooling: Max, Min (Shadow), MAD (Texture).
        x: (B, C, H, W)
        Returns: (B, 3*C)
        """
        # 1. Global Max Pooling
        # Flatten spatial dims to find max
        # x.view(B, C, -1).max(2).values is equivalent to global max pool
        max_pool = x.view(x.size(0), x.size(1), -1).max(dim=2)[0]

        # 2. Global Min Pooling (Shadow Depth)
        # Implemented as Max(-x) per description to capture magnitude of voids
        min_pool = (-x).view(x.size(0), x.size(1), -1).max(dim=2)[0]

        # 3. Global MAD Pooling (Mean Absolute Deviation)
        # Mean over spatial dimensions
        mean_val = x.mean(dim=(2, 3), keepdim=True)
        # MAD = Mean(|x - Mean|)
        mad_pool = (x - mean_val).abs().mean(dim=(2, 3))

        # Concatenate: (B, C) -> (B, 3*C)
        return torch.cat([max_pool, min_pool, mad_pool], dim=1)

    def forward(self, x_img, x_angle):
        # x_img: (B, 3, 75, 75)
        # x_angle: (B,) or (B, 1)

        # Ensure angle is (B, 1)
        if x_angle.dim() == 1:
            x_angle = x_angle.view(-1, 1)

        # Pass through backbone, collecting features
        features = []
        out = x_img

        for i, stage in enumerate(self.stages):
            out = stage(out)

            # If this stage is in our extraction list
            if i in self.extract_indices:
                # Find which projection to use (map index in extract_indices to index in projections)
                proj_idx = self.extract_indices.index(i)

                # Project: (B, 128, H, W) -> (B, 64, H, W)
                proj_feat = self.projections[proj_idx](out)

                # Pool: (B, 64, H, W) -> (B, 192)
                pooled_feat = self._tri_stat_pooling(proj_feat)

                features.append(pooled_feat)

        # Concatenate all stage features: (B, 384)
        global_features = torch.cat(features, dim=1)

        # Fuse with angle: (B, 385)
        # Angle is raw, not normalized
        combined = torch.cat([global_features, x_angle], dim=1)

        # Head
        logits = self.head(combined)

        # Squeeze output to (B,) if necessary, but BCEWithLogitsLoss often takes (B,1)
        # We return (B, 1) to be safe
        return logits
