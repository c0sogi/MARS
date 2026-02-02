import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeMPooling(nn.Module):
    """
    Generalized Mean Pooling (GeM)
    Computes the generalized mean: f(x) = (1/N * sum(x^p))^(1/p)

    This pooling layer is differentiable and allows the model to learn
    to focus on salient features (like max pooling) or aggregate context (like avg pooling).
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeMPooling, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x shape: (B, C, H, W)
        # Clamp to avoid NaN in pow (numerical stability)
        x = x.clamp(min=eps)

        # Average over spatial dimensions (H, W) -> dims (-2, -1)
        # Result shape: (B, C)
        x = x.pow(p).mean(dim=(-2, -1))

        # Root
        x = x.pow(1.0 / p)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class ConvNeXtGeM(nn.Module):
    """
    ConvNeXt-Tiny architecture modified with GeM Pooling.

    Key Features:
    1. Uses pre-trained ConvNeXt-Tiny backbone.
    2. Retains the original LayerNorm from the backbone.
    3. Applies LayerNorm BEFORE pooling (as per specific task requirement).
    4. Replaces Global Average Pooling with GeM Pooling.
    """

    def __init__(self, config=Config):
        super(ConvNeXtGeM, self).__init__()

        # Initialize backbone using timm
        # num_classes=0 removes the default fc head
        # global_pool='' removes the default pooling, returning spatial features (B, C, H, W)
        self.backbone = timm.create_model(
            config.MODEL_NAME,
            pretrained=config.PRETRAINED,
            num_classes=0,
            global_pool="",
            drop_path_rate=config.DROP_PATH_RATE,
        )

        # Extract feature dimension (768 for ConvNeXt-Tiny)
        self.num_features = self.backbone.num_features

        # Extract the normalization layer from the backbone
        # In timm's ConvNeXt, self.backbone.norm is the final LayerNorm layer
        if hasattr(self.backbone, "norm") and self.backbone.norm is not None:
            self.norm = self.backbone.norm
            # We remove it from the backbone execution flow to prevent double execution
            # (though forward_features usually skips it, this is for safety)
            self.backbone.norm = nn.Identity()
        else:
            # Fallback (standard ConvNeXt should have a norm)
            self.norm = nn.Identity()

        # GeM Pooling Layer
        self.gem = GeMPooling(p=config.GEM_P_INIT)

        # Classification Head
        self.head = nn.Linear(self.num_features, config.NUM_CLASSES)

        # Configuration flag
        self.use_norm_before_pool = config.LAYERNORM_BEFORE_POOLING

    def forward(self, x):
        # 1. Backbone Features
        # Returns spatial features: (B, C, H, W)
        x = self.backbone.forward_features(x)

        # 2. Normalization (LayerNorm)
        # Applied before pooling as per requirement.
        if self.use_norm_before_pool:
            # ConvNeXt's LayerNorm is typically nn.LayerNorm or compatible.
            # nn.LayerNorm expects input shape (..., C) (Channels Last).
            # Our features are (B, C, H, W) (Channels First).

            if isinstance(self.norm, nn.LayerNorm):
                # Permute to (B, H, W, C) for normalization
                x = x.permute(0, 2, 3, 1)
                x = self.norm(x)
                # Permute back to (B, C, H, W)
                x = x.permute(0, 3, 1, 2)
            else:
                # If it's a custom LayerNorm2d or Identity, apply directly
                x = self.norm(x)

        # 3. GeM Pooling
        # Reduces (B, C, H, W) -> (B, C)
        x = self.gem(x)

        # 4. Flatten
        # Ensures shape is (B, C) before the linear layer
        x = torch.flatten(x, 1)

        # 5. Classifier
        x = self.head(x)

        return x
