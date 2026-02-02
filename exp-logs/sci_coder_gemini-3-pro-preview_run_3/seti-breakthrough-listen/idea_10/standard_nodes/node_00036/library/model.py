import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class SiameseEfficientNet(nn.Module):
    """
    Siamese EfficientNet-B0 with Explicit Difference and Multi-View Pooling (GAP+GMP).

    Architecture:
    1. Shared EfficientNet-B0 backbone (features_only=True, out_indices=(4,)).
       Extracts 320-channel features from the final block (before projection).
    2. Explicit Spatial Difference: F_diff = F_on - F_off.
    3. Multi-View Pooling: Apply GAP and GMP to F_on, F_off, and F_diff.
    4. Concat (6 vectors) and Linear Head.
    """

    def __init__(self):
        super(SiameseEfficientNet, self).__init__()

        # Load backbone
        # Use features_only=True to get the output of the final block (320 channels)
        # rather than the projected output (1280 channels).
        # Cite solution_lesson_node_00035
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            in_chans=Config.IN_CHANNELS,
            features_only=True,
            out_indices=(4,),
        )

        # Determine feature channels
        # EfficientNet-B0 block 4 output typically has 320 channels
        dummy_input = torch.randn(
            1, Config.IN_CHANNELS, Config.IMG_HEIGHT, Config.IMG_WIDTH
        )
        with torch.no_grad():
            features = self.backbone(dummy_input)
            # features is a list, we take the last one
            self.feature_dim = features[-1].shape[1]

        # Classification Head
        # Input:
        #   F_on:   GAP + GMP -> 2 vectors
        #   F_off:  GAP + GMP -> 2 vectors
        #   F_diff: GAP + GMP -> 2 vectors
        # Total: 6 * feature_dim
        # Cite solution_lesson_node_00027, solution_lesson_node_00034
        self.fc = nn.Linear(self.feature_dim * 6, 1)

    def forward(self, x):
        # x is a tuple/list: (on_target_images, off_target_images)
        # Each shape: (Batch, 3, H, W)
        x_on, x_off = x

        # 1. Extract Features (Shared Backbone)
        # Returns list of feature maps, take the last one
        f_on = self.backbone(x_on)[-1]
        f_off = self.backbone(x_off)[-1]

        # 2. Explicit Spatial Difference
        # Cite solution_lesson_node_00022
        f_diff = f_on - f_off

        # 3. Multi-View Pooling (GAP + GMP)
        def pool(f):
            # Global Average Pooling
            gap = F.adaptive_avg_pool2d(f, (1, 1)).flatten(1)
            # Global Max Pooling
            gmp = F.adaptive_max_pool2d(f, (1, 1)).flatten(1)
            return gap, gmp

        gap_on, gmp_on = pool(f_on)
        gap_off, gmp_off = pool(f_off)
        gap_diff, gmp_diff = pool(f_diff)

        # 4. Concatenation
        # Shape: (Batch, feature_dim * 6)
        v_concat = torch.cat(
            [gap_on, gmp_on, gap_off, gmp_off, gap_diff, gmp_diff], dim=1
        )

        # 5. Classification
        output = self.fc(v_concat)

        return output
