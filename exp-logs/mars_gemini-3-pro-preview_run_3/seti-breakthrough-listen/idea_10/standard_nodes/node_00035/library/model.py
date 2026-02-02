import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class SiameseEfficientNet(nn.Module):
    """
    Siamese EfficientNet-B0 with Explicit GAP+GMP Pooling.

    Architecture:
    1. Shared EfficientNet-B0 backbone (ImageNet weights).
    2. Explicit Feature Difference: F_diff = F_on - F_off.
    3. GAP and GMP on F_on, F_off, and F_diff.
    4. Concat and Linear Head.
    """

    def __init__(self):
        super(SiameseEfficientNet, self).__init__()

        # Load backbone
        # We use features_only=False but will call forward_features manually
        # to get spatial maps (B, C, H, W)
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            in_chans=Config.IN_CHANNELS,
            num_classes=0,  # We don't need the original classifier
        )

        # Determine feature channels
        # EfficientNet-B0 typically has 1280 channels at the final layer
        dummy_input = torch.randn(
            1, Config.IN_CHANNELS, Config.IMG_HEIGHT, Config.IMG_WIDTH
        )
        with torch.no_grad():
            features = self.backbone.forward_features(dummy_input)
            self.feature_dim = features.shape[1]

        # Classification Head
        # Input: Concat of GAP and GMP for On, Off, Diff
        # 3 streams * 2 poolings = 6 vectors
        self.fc = nn.Linear(self.feature_dim * 6, 1)

    def forward(self, x):
        # x is a tuple/list: (on_target_images, off_target_images)
        # Each shape: (Batch, 3, H, W)
        x_on, x_off = x

        # 1. Extract Features (Shared Backbone)
        # Shape: (Batch, 1280, H', W')
        f_on = self.backbone.forward_features(x_on)
        f_off = self.backbone.forward_features(x_off)

        # 2. Explicit Feature Difference
        # Cite solution_lesson_node_00019: Explicit difference works better than learned fusion
        f_diff = f_on - f_off

        # 3. Pooling (GAP + GMP)
        # Cite solution_lesson_node_00034: Explicit Max+Avg pooling outperforms GeM
        def pool(f):
            # Global Average Pooling
            avg_p = F.adaptive_avg_pool2d(f, (1, 1)).flatten(1)
            # Global Max Pooling
            max_p = F.adaptive_max_pool2d(f, (1, 1)).flatten(1)
            return avg_p, max_p

        v_on_avg, v_on_max = pool(f_on)
        v_off_avg, v_off_max = pool(f_off)
        v_diff_avg, v_diff_max = pool(f_diff)

        # 4. Concatenation
        # Shape: (Batch, 1280 * 6)
        v_concat = torch.cat(
            [v_on_avg, v_on_max, v_off_avg, v_off_max, v_diff_avg, v_diff_max], dim=1
        )

        # 5. Classification
        output = self.fc(v_concat)

        return output
