import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DilatedResidualBlock(nn.Module):
    """
    A Residual Block with Dilated Convolutions and Group Normalization.
    Maintains full spatial resolution (padding = dilation).
    """

    def __init__(self, in_channels, out_channels, dilation, groups=8):
        super(DilatedResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.gn1 = nn.GroupNorm(groups, out_channels)
        self.act = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.gn2 = nn.GroupNorm(groups, out_channels)

        # Identity mapping adjustment if channel dimensions change
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.gn2(out)

        out += residual
        out = self.act(out)
        return out


class FusionBlock(nn.Module):
    """
    Fuses features from the deeper path with skip connections from the encoder.
    Uses Concatenation -> Conv -> GN -> ReLU.
    """

    def __init__(self, in_channels_deep, in_channels_skip, out_channels, groups=8):
        super(FusionBlock, self).__init__()

        total_in_channels = in_channels_deep + in_channels_skip

        self.conv = nn.Conv2d(
            total_in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.gn = nn.GroupNorm(groups, out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x_deep, x_skip):
        # Concatenate along channel dimension
        x = torch.cat([x_deep, x_skip], dim=1)
        x = self.conv(x)
        x = self.gn(x)
        x = self.act(x)
        return x


class FRDUNet(nn.Module):
    """
    Full-Resolution Dilated U-Net.

    Architecture:
    1. Projection: Compress Z-depth (65) -> Projection Channels.
    2. Encoder: Sequential Dilated Residual Blocks (r=1, 2, 4, 8, 16).
    3. Decoder: Iterative aggregation of deep features with encoder skip connections.
    4. Head: 1x1 Conv to binary logits.
    """

    def __init__(self):
        super(FRDUNet, self).__init__()

        # --- 1. Learnable 2.5D Projection ---
        # Compresses the Z-dimension (treated as input channels) into a feature space
        self.projection = nn.Sequential(
            nn.Conv2d(
                Config.Z_DEPTH, Config.PROJECTION_CHANNELS, kernel_size=1, bias=False
            ),
            nn.GroupNorm(Config.GROUP_NORM_GROUPS, Config.PROJECTION_CHANNELS),
            nn.ReLU(inplace=True),
        )

        # --- 2. Dilated Encoder (Backbone) ---
        self.encoder_blocks = nn.ModuleList()

        in_ch = Config.PROJECTION_CHANNELS
        out_ch = Config.BACKBONE_CHANNELS

        for dilation in Config.DILATION_RATES:
            block = DilatedResidualBlock(
                in_ch, out_ch, dilation=dilation, groups=Config.GROUP_NORM_GROUPS
            )
            self.encoder_blocks.append(block)
            # Subsequent blocks input channels match the backbone width
            in_ch = out_ch

        # --- 3. Iterative Aggregation Decoder ---
        self.decoder_blocks = nn.ModuleList()

        # We fuse from the second-to-last block upwards.
        # If we have N blocks, we have N-1 fusion steps.
        # Deep input is always BACKBONE_CHANNELS.
        # Skip input is always BACKBONE_CHANNELS (since all encoder blocks output this).

        num_fusion_steps = len(Config.DILATION_RATES) - 1

        for _ in range(num_fusion_steps):
            fusion = FusionBlock(
                in_channels_deep=Config.BACKBONE_CHANNELS,
                in_channels_skip=Config.BACKBONE_CHANNELS,
                out_channels=Config.BACKBONE_CHANNELS,
                groups=Config.GROUP_NORM_GROUPS,
            )
            self.decoder_blocks.append(fusion)

        # --- 4. Classification Head ---
        self.classifier = nn.Conv2d(Config.BACKBONE_CHANNELS, 1, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, Z, H, W) or (B, 1, Z, H, W).
               Z is expected to be Config.Z_DEPTH (65).
        """
        # Handle potential extra channel dim from dataloaders
        if x.dim() == 5:
            x = x.squeeze(1)

        # 1. Projection
        x = self.projection(x)  # (B, 32, H, W)

        # 2. Encoder Pass
        enc_features = []
        curr = x
        for block in self.encoder_blocks:
            curr = block(curr)
            enc_features.append(curr)

        # enc_features contains [feat_r1, feat_r2, feat_r4, feat_r8, feat_r16]

        # 3. Decoder Pass
        # Start with the deepest feature map
        curr_deep = enc_features[-1]

        # Iterate through fusion blocks
        # We need to match fusion blocks with skip connections in reverse order.
        # Fusion 0 uses skip -2 (second to last)
        # Fusion 1 uses skip -3
        # ...
        for i, fusion_block in enumerate(self.decoder_blocks):
            skip_idx = -2 - i
            skip_feat = enc_features[skip_idx]

            curr_deep = fusion_block(curr_deep, skip_feat)

        # 4. Classification
        logits = self.classifier(curr_deep)

        return logits
