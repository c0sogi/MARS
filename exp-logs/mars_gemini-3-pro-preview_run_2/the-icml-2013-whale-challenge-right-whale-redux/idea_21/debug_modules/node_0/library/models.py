import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes f(X) = (1/|X| * sum(x^p))^(1/p).
    This pooling strategy is effective for detecting transient signals (like whale calls)
    within a larger time-frequency representation, as it interpolates between
    Max Pooling (p -> infinity) and Average Pooling (p -> 1).
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x shape: (Batch, Channels, Height, Width)
        # 1. Clamp to avoid NaN in power operation
        # 2. Raise to power p
        # 3. Average pool over spatial dimensions (H, W)
        # 4. Raise to power 1/p
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )


class WhaleClassifier(nn.Module):
    """
    Whale Detection Model.
    Wraps a timm backbone (e.g., EfficientNet, ResNet) with specific adaptations:
    1. Input Layer: Adapted for 1-channel input (Spectrogram) via weight summation/averaging.
    2. Pooling: Uses GeM pooling instead of standard Global Average Pooling.
    3. Head: Simple Linear layer for binary classification.
    """

    def __init__(self, model_name, pretrained=True):
        """
        Args:
            model_name (str): Name of the timm model to load (e.g., 'tf_efficientnet_b0_ns').
            pretrained (bool): Whether to load ImageNet/NoisyStudent weights.
        """
        super(WhaleClassifier, self).__init__()

        # Load backbone from timm
        # in_chans=1: Automatically adapts the first conv layer weights from 3 channels to 1.
        # num_classes=0: Removes the default classification head.
        # global_pool='': Removes the default pooling, returning raw feature maps (B, C, H, W).
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=Config.IN_CHANNELS,
            num_classes=0,
            global_pool="",
        )

        # Determine the number of output channels from the backbone
        if hasattr(self.backbone, "num_features"):
            self.in_features = self.backbone.num_features
        else:
            # Fallback: Run a dummy forward pass to determine feature dimension
            with torch.no_grad():
                # Create a dummy input matching the expected input shape
                dummy = torch.randn(1, Config.IN_CHANNELS, 128, 128)
                features = self.backbone(dummy)
                self.in_features = features.shape[1]

        # Initialize Pooling Layer
        if Config.POOLING == "gem":
            self.pooling = GeM()
        else:
            # Fallback to standard Adaptive Average Pooling
            self.pooling = nn.AdaptiveAvgPool2d(1)

        # Classification Head
        # Projects features to the number of classes (1 for binary classification)
        self.head = nn.Linear(self.in_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 1, Freq, Time).

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes).
        """
        # 1. Extract Features
        # Shape: (Batch, Channels, Height, Width)
        features = self.backbone(x)

        # 2. Apply Pooling
        # Shape: (Batch, Channels, 1, 1)
        pooled = self.pooling(features)

        # 3. Flatten
        # Shape: (Batch, Channels)
        flattened = pooled.view(pooled.size(0), -1)

        # 4. Classification Head
        # Shape: (Batch, Num_Classes)
        logits = self.head(flattened)

        return logits
