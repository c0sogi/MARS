import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.

    Computes the generalized mean of the input tensor:
        f(x) = (mean(x^p))^(1/p)

    where p is a learnable parameter.
    - p -> 1: approaches Average Pooling
    - p -> infinity: approaches Max Pooling

    This is particularly useful for weakly supervised audio classification where
    the target signal (bird call) might be sparse within the spectrogram.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Clamp input to avoid numerical instability with pow
        # Apply average pooling on x^p
        # Then take the (1/p)-th power
        # x shape: (Batch, Channels, Height, Width)
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class BirdClassifier(nn.Module):
    """
    Bird Species Classifier using EfficientNet-B0 backbone and GeM Pooling.
    """

    def __init__(self, backbone=Config.BACKBONE, pretrained=Config.PRETRAINED):
        super(BirdClassifier, self).__init__()

        # Create backbone
        # num_classes=0 and global_pool="" ensures we get the feature maps
        # (Batch, C, H, W) instead of pooled vectors.
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Determine the number of input features for the head
        self.in_features = self.backbone.num_features

        # Pooling Layer
        if Config.USE_GEM_POOLING:
            self.pooling = GeM()
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)

        # Classification Head
        self.fc = nn.Linear(self.in_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input spectrograms of shape (Batch, 3, H, W).

        Returns:
            torch.Tensor: Logits of shape (Batch, NumClasses).
        """
        # Extract features from backbone
        # Shape: (Batch, Channels, H_feat, W_feat)
        x = self.backbone(x)

        # Apply pooling
        # Shape: (Batch, Channels, 1, 1)
        x = self.pooling(x)

        # Flatten
        # Shape: (Batch, Channels)
        x = x.flatten(1)

        # Classification head
        # Shape: (Batch, NumClasses)
        logits = self.fc(x)

        return logits
