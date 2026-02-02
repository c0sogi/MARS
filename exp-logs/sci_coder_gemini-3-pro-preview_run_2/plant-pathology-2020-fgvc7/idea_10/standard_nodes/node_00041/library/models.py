import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import CFG


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes f(x) = (1/N * sum(x^p))^(1/p).
    This acts as a soft transition between Average Pooling (p=1) and Max Pooling (p -> infinity),
    allowing the model to focus on localized salient features (lesions) while retaining context.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

        # Check configuration for trainability of p
        if hasattr(CFG, "gem_trainable") and not CFG.gem_trainable:
            self.p.requires_grad = False

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x shape: (B, C, H, W)
        # Clamp input to avoid NaN gradients with power operation
        x = torch.clamp(x, min=eps)

        # 1. Raise to power p
        # 2. Average pool over spatial dimensions (H, W) -> Result (B, C, 1, 1)
        # 3. Raise to power 1/p
        return F.avg_pool2d(x.pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class AppleNet(nn.Module):
    """
    Apple Disease Detection Model.

    Architecture:
    1. Backbone: EfficientNetV2-L or ConvNeXt-Base (via timm).
    2. Pooling: Generalized Mean (GeM) Pooling to capture localized disease artifacts.
    3. Head: Multi-label decomposition (2 outputs: Rust, Scab).
    """

    def __init__(self, model_name, pretrained=True):
        super(AppleNet, self).__init__()

        # Load the backbone from timm
        # num_classes=0 and global_pool='' ensures we get the raw spatial feature maps (B, C, H, W)
        # instead of a pooled vector or classification logits.
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Retrieve the number of channels in the feature map
        self.in_features = self.backbone.num_features

        # Initialize Generalized Mean Pooling
        # p is initialized to CFG.gem_p (default 3.0)
        self.gem = GeM(p=CFG.gem_p)

        # Classification Head
        # We output logits for 'rust' and 'scab'.
        # 'Healthy' and 'Multiple' are derived from these two binary predictions.
        self.head = nn.Linear(self.in_features, len(CFG.target_cols))

    def forward(self, x):
        # Extract spatial features from backbone
        # Shape: (Batch_Size, Channels, Height, Width)
        features = self.backbone(x)

        # Apply GeM Pooling
        # Shape: (Batch_Size, Channels, 1, 1)
        pooled = self.gem(features)

        # Flatten the pooled features
        # Shape: (Batch_Size, Channels)
        pooled = pooled.flatten(1)

        # Pass through classification head
        # Shape: (Batch_Size, 2)
        logits = self.head(pooled)

        return logits
