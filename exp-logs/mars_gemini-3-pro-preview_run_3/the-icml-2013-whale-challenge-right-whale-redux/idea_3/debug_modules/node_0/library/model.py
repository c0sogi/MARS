import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the generalized mean: (1/N * sum(x^p))^(1/p)
    """

    def __init__(self, p=3.0, eps=1e-6, trainable=True):
        super(GeM, self).__init__()
        # Initialize p. If trainable, it is an nn.Parameter.
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.p.requires_grad = trainable

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)

        # Clamp inputs to avoid numerical instability with pow()
        x = x.clamp(min=self.eps)

        # Calculate spatial dimensions for average pooling
        # We want to pool over the spatial dimensions (H, W)
        h, w = x.size(-2), x.size(-1)

        # Formula: (AvgPool(x^p))^(1/p)
        # 1. Raise to power p
        x_p = x.pow(self.p)

        # 2. Average Pool over spatial dimensions
        avg_x_p = F.avg_pool2d(x_p, kernel_size=(h, w))

        # 3. Raise to power 1/p
        return avg_x_p.pow(1.0 / self.p)

    def __repr__(self):
        return f"GeM(p={self.p.data.item():.4f}, eps={self.eps}, trainable={self.p.requires_grad})"


class EfficientNetGeM(nn.Module):
    """
    EfficientNet-B0 model with Generalized Mean (GeM) Pooling.
    """

    def __init__(self, pretrained=Config.PRETRAINED):
        super(EfficientNetGeM, self).__init__()

        # Load EfficientNet-B0 backbone from timm
        # in_chans=1: Adapts the first convolution to accept 1-channel input (spectrogram)
        # num_classes=0, global_pool='': Removes the default classifier and pooling, returning feature maps
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=pretrained,
            in_chans=Config.IN_CHANNELS,
            num_classes=0,
            global_pool="",
        )

        # Retrieve the number of feature channels output by the backbone
        self.in_features = self.backbone.num_features

        # Initialize GeM Pooling
        self.pooling = GeM(p=Config.GEM_P, trainable=Config.GEM_TRAINABLE)

        # Final Classification Head
        self.fc = nn.Linear(self.in_features, Config.NUM_CLASSES)

    def forward(self, x):
        # Input x: (Batch, 1, Freq, Time)

        # Extract features
        # Output: (Batch, Channels, F', T')
        features = self.backbone(x)

        # Apply GeM Pooling
        # Output: (Batch, Channels, 1, 1)
        pooled = self.pooling(features)

        # Flatten
        # Output: (Batch, Channels)
        flattened = pooled.view(pooled.size(0), -1)

        # Classification
        # Output: (Batch, Num_Classes)
        logits = self.fc(flattened)

        return logits
