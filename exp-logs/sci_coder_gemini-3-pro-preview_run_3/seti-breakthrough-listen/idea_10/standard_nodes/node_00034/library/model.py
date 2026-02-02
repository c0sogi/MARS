import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the generalized mean of the spatial dimensions.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        # Clamp min value to eps to avoid numerical instability with power
        x = torch.clamp(x, min=self.eps)

        # Average over spatial dimensions (H, W) -> (Batch, Channels)
        # Formula: (mean(x^p))^(1/p)
        return (
            F.avg_pool2d(x.pow(self.p), (x.size(-2), x.size(-1)))
            .pow(1.0 / self.p)
            .squeeze(-1)
            .squeeze(-1)
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class SiameseEfficientNet(nn.Module):
    """
    Siamese EfficientNet-B0 with Adaptive Difference and GeM Pooling.

    Architecture:
    1. Shared EfficientNet-B0 backbone (ImageNet weights).
    2. Adaptive Feature Difference: F_diff = F_on - (w * F_off).
    3. GeM Pooling on F_on, F_off, and F_diff.
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

        # Adaptive Scaling Vector for Difference Module
        # Shape: (1, C, 1, 1) for broadcasting
        self.scale_vector = nn.Parameter(torch.ones(1, self.feature_dim, 1, 1))

        # Generalized Mean Pooling
        self.gem = GeM(p=Config.GEM_P)

        # Classification Head
        # Input: Concat of GeM(F_on), GeM(F_off), GeM(F_diff) -> 3 * feature_dim
        self.fc = nn.Linear(self.feature_dim * 3, 1)

    def forward(self, x):
        # x is a tuple/list: (on_target_images, off_target_images)
        # Each shape: (Batch, 3, H, W)
        x_on, x_off = x

        # 1. Extract Features (Shared Backbone)
        # Shape: (Batch, 1280, H', W')
        f_on = self.backbone.forward_features(x_on)
        f_off = self.backbone.forward_features(x_off)

        # 2. Adaptive Feature Difference
        # F_diff = F_on - (w * F_off)
        # The scale_vector adapts the noise floor of the off-target stream
        f_diff = f_on - (self.scale_vector * f_off)

        # 3. GeM Pooling
        # Each results in (Batch, 1280)
        v_on = self.gem(f_on)
        v_off = self.gem(f_off)
        v_diff = self.gem(f_diff)

        # 4. Concatenation
        # Shape: (Batch, 1280 * 3)
        v_concat = torch.cat([v_on, v_off, v_diff], dim=1)

        # 5. Classification
        output = self.fc(v_concat)

        return output
