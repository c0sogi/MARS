import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DropBlock2D(nn.Module):
    """
    DropBlock regularization: Drops contiguous regions of the feature map.
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # Only apply during training and if drop_prob > 0
        if not self.training or self.drop_prob == 0.0:
            return x

        N, C, H, W = x.shape

        # Ensure block_size fits within the feature map
        block_size = min(self.block_size, H, W)

        # Calculate gamma (probability of a pixel being a block center)
        # gamma = (drop_prob / block_size^2) * (feat_area / valid_area)
        feat_area = H * W
        valid_area = (H - block_size + 1) * (W - block_size + 1)
        gamma = (self.drop_prob / (block_size**2)) * (feat_area / valid_area)

        # Create a mask of zeros
        mask = torch.zeros((N, 1, H, W), device=x.device)

        # Define valid region for block centers
        pad_h = block_size // 2
        pad_w = block_size // 2

        # Sample seeds in the valid region
        # Note: We use the valid slice logic. If block_size is 5, pad is 2.
        # Valid centers are from index 2 to H-2 (exclusive of H-2? No, H-1-2).
        # Slicing in Python is start:stop.
        # We want indices [pad, H-pad). Length = H - 2*pad.
        # Check consistency with valid_area calculation:
        # If H=9, block=5, pad=2. Valid indices: 2, 3, 4, 5, 6. Count=5.
        # valid_area calc: 9-5+1 = 5. Matches.

        valid_mask_slice = torch.bernoulli(
            torch.full((N, 1, H - 2 * pad_h, W - 2 * pad_w), gamma, device=x.device)
        )
        mask[:, :, pad_h : H - pad_h, pad_w : W - pad_w] = valid_mask_slice

        # Expand seeds to blocks using MaxPool
        # Padding ensures the output size remains HxW
        mask = F.max_pool2d(
            mask,
            kernel_size=(block_size, block_size),
            stride=(1, 1),
            padding=(pad_h, pad_w),
        )

        # Handle edge cases where padding might slightly alter size if even block_size (though config uses 5)
        mask = mask[:, :, :H, :W]

        # Invert mask: 1 becomes 0 (drop), 0 becomes 1 (keep)
        block_mask = 1 - mask

        # Normalize the features to maintain expected activation scale
        # Scale = total_elements / kept_elements
        normalize_scale = block_mask.numel() / (block_mask.sum() + 1e-6)

        return x * block_mask * normalize_scale


class NBAModule(nn.Module):
    """
    Non-Bottleneck Attention (NBA) Module.
    Uses a full-rank MLP (no bottleneck) to model global channel dependencies.
    """

    def __init__(self, channels):
        super(NBAModule, self).__init__()
        # MLP: Linear -> LeakyReLU -> Linear -> Sigmoid
        # Maintains channel dimension (C -> C -> C)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LeakyReLU(Config.LEAKY_RELU_SLOPE, inplace=True),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Global Average Pooling -> (B, C)
        y = x.view(b, c, -1).mean(dim=2)
        # Excitation
        y = self.mlp(y)
        # Reshape to (B, C, 1, 1) and scale input
        y = y.view(b, c, 1, 1)
        return x * y


class DPDB_NBA_CNN(nn.Module):
    """
    Dual-Polarity DropBlock CNN with Non-Bottleneck Attention.
    """

    def __init__(self):
        super(DPDB_NBA_CNN, self).__init__()

        channels = Config.BACKBONE_CHANNELS  # [64, 128, 128, 128]
        in_c = Config.INPUT_CHANNELS

        # --- Stage 1 ---
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_c, channels[0], kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(Config.LEAKY_RELU_SLOPE, inplace=True),
            NBAModule(channels[0]),
            nn.MaxPool2d(2, 2),
        )

        # --- Stage 2 ---
        self.stage2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(channels[1]),
            nn.LeakyReLU(Config.LEAKY_RELU_SLOPE, inplace=True),
            NBAModule(channels[1]),
            nn.MaxPool2d(2, 2),
        )

        # --- Stage 3 ---
        # Note: DropBlock is applied before pooling
        self.stage3_block = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(channels[2]),
            nn.LeakyReLU(Config.LEAKY_RELU_SLOPE, inplace=True),
            NBAModule(channels[2]),
        )
        self.stage3_drop = (
            DropBlock2D(Config.DROPBLOCK_PROB, Config.DROPBLOCK_BLOCK_SIZE)
            if Config.USE_DROPBLOCK
            else nn.Identity()
        )
        self.stage3_pool = nn.MaxPool2d(2, 2)

        # --- Stage 4 ---
        self.stage4_block = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(channels[3]),
            nn.LeakyReLU(Config.LEAKY_RELU_SLOPE, inplace=True),
            NBAModule(channels[3]),
        )
        self.stage4_drop = (
            DropBlock2D(Config.DROPBLOCK_PROB, Config.DROPBLOCK_BLOCK_SIZE)
            if Config.USE_DROPBLOCK
            else nn.Identity()
        )
        self.stage4_pool = nn.MaxPool2d(2, 2)

        # --- Classification Head ---
        # Input: S3_Max(128) + S3_Min(128) + S4_Max(128) + S4_Min(128) + Angle(1) = 513
        head_in_dim = channels[2] * 2 + channels[3] * 2 + 1
        hidden_dim = 256

        self.head = nn.Sequential(
            nn.Linear(head_in_dim, hidden_dim),
            nn.LeakyReLU(Config.LEAKY_RELU_SLOPE, inplace=True),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, angle):
        # Stage 1
        x = self.stage1(x)

        # Stage 2
        x = self.stage2(x)

        # Stage 3
        x = self.stage3_block(x)
        x = self.stage3_drop(x)
        x = self.stage3_pool(x)
        s3_feat = x  # Capture Stage 3 output

        # Stage 4
        x = self.stage4_block(x)
        x = self.stage4_drop(x)
        x = self.stage4_pool(x)
        s4_feat = x  # Capture Stage 4 output

        # Dual-Polarity Pooling
        # Function to extract max and min
        def get_stats(feat_map):
            # Flatten spatial dims: (B, C, H, W) -> (B, C, H*W)
            flat = feat_map.view(feat_map.size(0), feat_map.size(1), -1)
            # Max and Min per channel
            max_val = flat.max(dim=2)[0]
            min_val = flat.min(dim=2)[0]
            return max_val, min_val

        s3_max, s3_min = get_stats(s3_feat)
        s4_max, s4_min = get_stats(s4_feat)

        # Reshape angle for concatenation
        angle = angle.view(-1, 1)

        # Fusion
        fused = torch.cat([s3_max, s3_min, s4_max, s4_min, angle], dim=1)

        # Classification
        out = self.head(fused)

        return out
