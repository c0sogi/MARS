import torch
import torch.nn as nn
from library.config import Config


class SpatiallyGatedBlock(nn.Module):
    """
    A residual block that explicitly decouples feature extraction from spatial relevance.

    Components:
    1. Feature Branch: Dilated Conv -> GroupNorm -> GeLU
       Extracts texture and shape features.
    2. Gating Branch: 1x1 Conv -> Sigmoid
       Produces a spatial attention map indicating ink probability.
    3. Fusion: (Feature * Gate) + Input
       Suppresses background noise (fibers) that matches texture but not structure.
    """

    def __init__(self, channels, dilation, groups):
        super(SpatiallyGatedBlock, self).__init__()

        # Feature Branch: standard dilated convolution for context
        self.feature_branch = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,  # Bias handled by GroupNorm
            ),
            nn.GroupNorm(num_groups=groups, num_channels=channels),
            nn.GELU(),
        )

        # Gating Branch: lightweight spatial attention
        self.gating_branch = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                padding=0,
                bias=True,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feature = self.feature_branch(x)
        gate = self.gating_branch(x)
        # Gated fusion with residual connection
        return (feature * gate) + x


class SGDN(nn.Module):
    """
    Spatially Gated Dilated Network (SGDN).

    Architecture:
    1. Input Projection: Learnable 2.5D projection (65 -> 32 channels).
    2. Backbone: Stack of SpatiallyGatedBlocks with sequential dilation rates.
    3. Head: 1x1 Convolution to produce binary logits.
    """

    def __init__(self):
        super(SGDN, self).__init__()

        # Load architecture hyperparameters from Config
        self.z_dim = Config.Z_DIM
        self.base_channels = Config.BASE_CHANNELS
        self.dilation_rates = Config.DILATION_RATES
        self.groups = Config.GROUP_NORM_GROUPS

        # 1. Learnable 2.5D Projection (Input Stage)
        # Projects the 65-channel volumetric input into a compact feature space.
        # This acts as a "soft" slice selection mechanism.
        self.projection = nn.Conv2d(
            in_channels=self.z_dim,
            out_channels=self.base_channels,
            kernel_size=1,
            bias=False,
        )

        # 2. Spatially Gated Residual Backbone
        # Constructs the stack of blocks based on the dilation schedule.
        layers = []
        for d in self.dilation_rates:
            layers.append(
                SpatiallyGatedBlock(
                    channels=self.base_channels, dilation=d, groups=self.groups
                )
            )
        self.backbone = nn.Sequential(*layers)

        # 3. Classification Head
        # Maps features to a single channel logit map.
        self.head = nn.Conv2d(
            in_channels=self.base_channels, out_channels=1, kernel_size=1, bias=True
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input volume tensor of shape (B, Z, H, W).
                              Z corresponds to the depth (65 slices).
        Returns:
            torch.Tensor: Output logits of shape (B, 1, H, W).
        """
        # Project Z-dimension to feature channels
        x = self.projection(x)

        # Pass through the gated backbone
        x = self.backbone(x)

        # Generate logits
        logits = self.head(x)

        return logits
