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


class PathologyModel(nn.Module):
    """
    Wrapper for ConvNeXt-Tiny that supports both GeM and standard Global Average Pooling (GAP).
    """

    def __init__(self, config=Config):
        super(PathologyModel, self).__init__()
        self.use_gem = config.USE_GEM_POOLING

        if self.use_gem:
            # Initialize backbone for GeM (remove head and pooling)
            self.backbone = timm.create_model(
                config.MODEL_NAME,
                pretrained=config.PRETRAINED,
                num_classes=0,
                global_pool="",
                drop_path_rate=config.DROP_PATH_RATE,
            )

            # Extract feature dimension
            self.num_features = self.backbone.num_features

            # Handle Normalization for GeM
            if hasattr(self.backbone, "norm") and self.backbone.norm is not None:
                self.norm = self.backbone.norm
                self.backbone.norm = nn.Identity()
            else:
                self.norm = nn.Identity()

            self.gem = GeMPooling(p=config.GEM_P_INIT)
            self.head = nn.Linear(self.num_features, config.NUM_CLASSES)
            self.use_norm_before_pool = config.LAYERNORM_BEFORE_POOLING

        else:
            # Standard timm model with GAP (Cite Lesson 00015)
            # num_classes=1 ensures proper Head (Norm + Linear) is created (Cite Lesson 00013)
            self.model = timm.create_model(
                config.MODEL_NAME,
                pretrained=config.PRETRAINED,
                num_classes=config.NUM_CLASSES,
                drop_path_rate=config.DROP_PATH_RATE,
            )

    def forward(self, x):
        if not self.use_gem:
            return self.model(x)

        # GeM Forward Pass
        x = self.backbone.forward_features(x)

        if self.use_norm_before_pool:
            if isinstance(self.norm, nn.LayerNorm):
                x = x.permute(0, 2, 3, 1)
                x = self.norm(x)
                x = x.permute(0, 3, 1, 2)
            else:
                x = self.norm(x)

        x = self.gem(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x
