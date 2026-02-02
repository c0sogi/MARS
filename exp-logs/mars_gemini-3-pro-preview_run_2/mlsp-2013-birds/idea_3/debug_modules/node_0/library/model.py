import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.

    Args:
        p (float): Initial value for the power parameter.
        eps (float): Small value for numerical stability.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        # Clamp min value to eps to avoid numerical issues with pow
        x = x.clamp(min=self.eps).pow(self.p)

        # Apply average pooling over the spatial dimensions (Height, Width)
        # Output shape: (Batch, Channels, 1, 1)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))

        # Raise to the power of 1/p
        x = x.pow(1.0 / self.p)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class BirdClassifier(nn.Module):
    """
    Bird Species Classifier using an EfficientNet backbone and GeM pooling.

    Args:
        backbone (str): Name of the timm backbone (default: efficientnet_b0).
        pretrained (bool): Whether to load pretrained ImageNet weights.
        num_classes (int): Number of output classes.
    """

    def __init__(
        self,
        backbone=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
    ):
        super(BirdClassifier, self).__init__()

        # Create backbone model using timm
        # num_classes=0 removes the final classification layer
        # global_pool='' removes the default pooling layer so we can use our own
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Get the number of features output by the backbone
        in_features = self.backbone.num_features

        # Pooling layer
        if Config.USE_GEM_POOLING:
            self.global_pool = GeM()
        else:
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Final classification head
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Extract features from backbone
        # Shape: (Batch, Channels, H, W)
        features = self.backbone(x)

        # Apply pooling
        # Shape: (Batch, Channels, 1, 1)
        pooled = self.global_pool(features)

        # Flatten
        # Shape: (Batch, Channels)
        pooled = pooled.flatten(1)

        # Classification
        # Shape: (Batch, Num_Classes)
        logits = self.fc(pooled)

        return logits
