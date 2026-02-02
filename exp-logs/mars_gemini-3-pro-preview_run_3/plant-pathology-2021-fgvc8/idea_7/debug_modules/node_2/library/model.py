import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes the generalized mean of the spatial features, allowing the model
    to focus on salient regions (high activations) similar to Max Pooling,
    or global context similar to Average Pooling, controlled by a learnable parameter p.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)

        # Clamp input to ensure non-negative values (or at least > eps) for power operation
        # and numerical stability.
        x = x.clamp(min=self.eps)

        # Calculate x^p
        x_pow = x.pow(self.p)

        # Apply Average Pooling on the spatial dimensions (H, W)
        # This computes 1/N * sum(x^p)
        avg_pool = F.avg_pool2d(x_pow, (x.size(-2), x.size(-1)))

        # Raise to the power of 1/p to complete the generalized mean formula
        gem_pool = avg_pool.pow(1.0 / self.p)

        return gem_pool


class AppleDiseaseModel(nn.Module):
    """
    Apple Disease Detection Model.

    Architecture:
    1. Backbone: ConvNeXt-Small (from timm)
    2. Pooling: Generalized Mean Pooling (GeM)
    3. Head: Linear Classification Layer
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        num_classes=Config.NUM_CLASSES,
        pretrained=Config.PRETRAINED,
    ):
        super(AppleDiseaseModel, self).__init__()

        # Initialize Backbone
        # num_classes=0 and global_pool="" are used to remove the default classifier
        # and pooling, returning the raw spatial feature maps (B, C, H, W).
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            drop_path_rate=Config.DROP_PATH_RATE,
        )

        # Determine the number of input features for the classification head
        if hasattr(self.backbone, "num_features"):
            self.in_features = self.backbone.num_features
        else:
            # Fallback: Run a dummy forward pass to infer shape
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
                features = self.backbone(dummy_input)
                self.in_features = features.shape[1]

        # Initialize Custom Pooling
        self.pooling = GeM()

        # Initialize Classification Head
        self.flatten = nn.Flatten()
        self.head = nn.Linear(self.in_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input image tensor of shape (Batch, 3, Height, Width).

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes).
        """
        # 1. Backbone Feature Extraction
        # Output shape: (Batch, Channels, H_feat, W_feat)
        features = self.backbone(x)

        # 2. GeM Pooling
        # Output shape: (Batch, Channels, 1, 1)
        pooled = self.pooling(features)

        # 3. Flatten
        # Output shape: (Batch, Channels)
        flattened = self.flatten(pooled)

        # 4. Classification Head
        # Output shape: (Batch, Num_Classes)
        logits = self.head(flattened)

        return logits
