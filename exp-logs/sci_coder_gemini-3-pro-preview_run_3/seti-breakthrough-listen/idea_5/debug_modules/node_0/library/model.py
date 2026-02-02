import torch
import torch.nn as nn
import timm
from library.utils import Config


class SiameseSpatialFusionNet(nn.Module):
    """
    Siamese Spatial-Fusion Network using EfficientNet-B0 backbone.

    Architecture:
    1. Shared Backbone: EfficientNet-B0 (ImageNet weights) extracts spatial features
       from 'On' and 'Off' streams independently.
    2. Spatial Fusion: Feature maps are concatenated along the channel dimension.
    3. Interaction Block: 1x1 Conv (bottleneck) -> 3x3 Conv (spatial mixing).
    4. Classification Head: Global Average Pooling -> Linear.
    """

    def __init__(self):
        super(SiameseSpatialFusionNet, self).__init__()

        # 1. Siamese Backbone
        # Load EfficientNet-B0 with ImageNet weights.
        # num_classes=0 and global_pool='' ensures we get the final feature maps (N, C, H, W)
        # instead of a pooled vector.
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="",
        )

        # Determine backbone output channels (1280 for EfficientNet-B0)
        if hasattr(self.backbone, "num_features"):
            self.backbone_dim = self.backbone.num_features
        else:
            # Fallback for standard EfficientNet-B0 if attribute is missing
            self.backbone_dim = 1280

        # 2. Spatial Fusion & Interaction Block
        # Input: Concatenation of On and Off feature maps (2 * backbone_dim)
        fusion_input_dim = self.backbone_dim * 2
        inter_dim = 512  # Bottleneck dimension

        self.interaction_block = nn.Sequential(
            # 1x1 Conv to reduce dimensionality (Channel interaction)
            nn.Conv2d(fusion_input_dim, inter_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_dim),
            nn.ReLU(inplace=True),
            # 3x3 Conv for local spatial comparison
            nn.Conv2d(inter_dim, inter_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inter_dim),
            nn.ReLU(inplace=True),
        )

        # 3. Classification Head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(inter_dim, 1)

    def forward(self, x_on, x_off):
        """
        Forward pass of the Siamese Network.

        Args:
            x_on (torch.Tensor): Batch of On-Target images (N, 3, H, W)
            x_off (torch.Tensor): Batch of Off-Target images (N, 3, H, W)

        Returns:
            torch.Tensor: Logits (N, 1)
        """
        # Pass through shared backbone
        f_on = self.backbone(x_on)  # Shape: (N, 1280, H', W')
        f_off = self.backbone(x_off)  # Shape: (N, 1280, H', W')

        # Spatial Fusion: Concatenate along channel dimension
        f_fused = torch.cat([f_on, f_off], dim=1)  # Shape: (N, 2560, H', W')

        # Interaction Block
        x = self.interaction_block(f_fused)  # Shape: (N, 512, H', W')

        # Classification
        x = self.global_pool(x)  # Shape: (N, 512, 1, 1)
        x = x.flatten(1)  # Shape: (N, 512)
        logits = self.classifier(x)  # Shape: (N, 1)

        return logits
