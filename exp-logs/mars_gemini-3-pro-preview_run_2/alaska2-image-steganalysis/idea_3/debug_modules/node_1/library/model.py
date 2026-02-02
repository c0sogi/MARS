import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from library.config import Config
from library.srm_filters import get_srm_layer


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the generalized mean of each channel in the feature map.
    Formula: f(X) = (1/|X| * sum(x^p))^(1/p)
    """

    def __init__(self, p=3.0, eps=1e-6, trainable=True):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p, requires_grad=trainable)
        self.eps = eps

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3.0, eps=1e-6):
        # Clamp input to epsilon to avoid NaN gradients in power operation
        x = x.clamp(min=eps)
        # Calculate average of x^p over spatial dimensions (H, W)
        x_pow = x.pow(p)
        avg_x_pow = F.avg_pool2d(x_pow, (x.size(-2), x.size(-1)))
        # Take the p-th root
        return avg_x_pow.pow(1.0 / p)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class ResV2GeM(nn.Module):
    """
    Optimized Residual-V2 Network (ResV2-GeM) for Steganography Detection.

    Architecture Pipeline:
    1. Fixed SRM Residual Stem (30 filters) -> Extracts noise residuals.
    2. Learnable Projection (30 -> 3 channels) -> Adapts residuals for backbone.
    3. EfficientNetV2-Small Backbone -> Extracts high-level features.
    4. GeM Pooling -> Focuses on high-activation regions (stego artifacts).
    5. Linear Classifier -> Binary prediction.
    """

    def __init__(self, config=Config, pretrained=True):
        super(ResV2GeM, self).__init__()

        # 1. SRM Residual Stem
        # Fixed, non-trainable layer of 30 high-pass filters derived from Spatial Rich Models
        self.srm = get_srm_layer()

        # 2. Learnable Input Projection
        # Projects 30-channel residuals down to 3 channels to match backbone input expectation.
        # Kernel size 3x3 preserves local spatial context.
        self.projection = nn.Conv2d(
            in_channels=30, out_channels=3, kernel_size=3, padding=1, bias=False
        )

        # Initialize projection weights to facilitate convergence
        nn.init.kaiming_normal_(
            self.projection.weight, mode="fan_out", nonlinearity="relu"
        )

        # 3. Backbone (EfficientNetV2-Small)
        # We use num_classes=0 to initialize without the top classification layer.
        self.backbone = timm.create_model(
            config.model_name,
            pretrained=pretrained,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
            num_classes=0,
        )

        # Retrieve the number of output features from the backbone (e.g., 1792 for efficientnetv2_rw_s)
        self.num_features = self.backbone.num_features

        # 4. GeM Pooling
        if config.use_gem:
            self.global_pool = GeM(p=config.gem_p_init, trainable=config.gem_trainable)
        else:
            # Fallback to standard GAP if GeM is disabled
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 5. Classifier Head
        self.fc = nn.Linear(self.num_features, config.target_size)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input images of shape (Batch, 3, Height, Width).

        Returns:
            torch.Tensor: Raw logits of shape (Batch, target_size).
        """
        # 1. Extract Residuals: (B, 3, H, W) -> (B, 30, H, W)
        x = self.srm(x)

        # 2. Project to Backbone Input: (B, 30, H, W) -> (B, 3, H, W)
        x = self.projection(x)

        # 3. Backbone Features: (B, 3, H, W) -> (B, C, H', W')
        # forward_features returns the unpooled feature maps
        x = self.backbone.forward_features(x)

        # 4. Pooling: (B, C, H', W') -> (B, C, 1, 1)
        x = self.global_pool(x)

        # Flatten: (B, C, 1, 1) -> (B, C)
        x = x.flatten(1)

        # 5. Classification: (B, C) -> (B, target_size)
        x = self.fc(x)

        return x
