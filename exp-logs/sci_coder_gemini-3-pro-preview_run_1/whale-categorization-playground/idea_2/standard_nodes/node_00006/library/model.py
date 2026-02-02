import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes the generalized mean: f(X) = (1/|X| * sum(x^p))^(1/p)

    Args:
        p (float): Initial value for the power parameter.
        eps (float): Small constant for numerical stability.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)

        # Clamp input to avoid numerical instability (e.g., log of 0 or negative bases if p is float)
        # Since we are dealing with ReLU outputs from EfficientNet, values are >= 0.
        x = torch.clamp(x, min=self.eps)

        # 1. Raise to power p
        x_pow = x.pow(self.p)

        # 2. Global Average Pooling (spatial dimensions)
        # Using adaptive_avg_pool2d with output size (1, 1) effectively computes the mean
        pooled = F.adaptive_avg_pool2d(x_pow, (1, 1))

        # 3. Raise to power 1/p
        gem_out = pooled.pow(1.0 / self.p)

        return gem_out

    def __repr__(self):
        return f"GeM(p={self.p.data.item():.4f}, eps={self.eps})"


class WhaleEfficientNet(nn.Module):
    """
    Fine-Grained Whale Species Classifier.
    Backbone: EfficientNet-B2
    Pooling: GeM
    Head: Linear
    """

    def __init__(self, model_name=None, pretrained=True, num_classes=None):
        super(WhaleEfficientNet, self).__init__()

        # Use Config defaults if arguments are not provided
        if model_name is None:
            model_name = Config.BACKBONE
        if num_classes is None:
            num_classes = Config.NUM_CLASSES

        # Load the backbone using timm
        # num_classes=0 and global_pool='' tells timm to return the raw feature map
        # instead of the pooled vector or logits.
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Determine the number of input features for the head
        # timm models usually have a num_features attribute
        in_features = self.backbone.num_features

        # Initialize Generalized Mean Pooling
        # Initial p value is taken from Config
        self.pooling = GeM(p=Config.GEM_P)

        # Classification Head
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W)

        Returns:
            logits (torch.Tensor): Class logits of shape (B, num_classes)
        """
        # 1. Feature Extraction
        features = self.backbone(x)  # Output: (B, C, H_feat, W_feat)

        # 2. GeM Pooling
        pooled = self.pooling(features)  # Output: (B, C, 1, 1)

        # 3. Flatten
        flattened = pooled.view(pooled.size(0), -1)  # Output: (B, C)

        # 4. Classification
        logits = self.head(flattened)  # Output: (B, num_classes)

        return logits
