import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.

    Formula: f(X) = (1/|X| * sum(x^p))^(1/p)

    - p -> 1: Average Pooling
    - p -> infinity: Max Pooling

    This pooling layer is differentiable and allows the model to learn
    the best pooling strategy for the specific task.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter initialized to 3.0
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x shape: (Batch, Channels, Height, Width)
        # Clamp min value to eps to avoid numerical instability with pow
        # Apply average pooling on x^p over the spatial dimensions (H, W)
        # Raise result to 1/p
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class AppleClassifier(nn.Module):
    """
    Apple Disease Detection Model.

    Wraps a timm backbone (e.g., ConvNeXt, MaxViT) and replaces the
    classification head with a GeM Pooling layer and a Linear layer.
    """

    def __init__(self, model_name: str, pretrained: bool = True):
        """
        Args:
            model_name (str): Name of the timm model to load.
            pretrained (bool): Whether to load pretrained weights.
        """
        super(AppleClassifier, self).__init__()

        # Create the backbone model
        # num_classes=0 removes the default linear head
        # global_pool="" removes the default pooling layer, returning feature maps (B, C, H, W)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Determine the number of input features for the head
        # We use num_features if available, otherwise infer from a dummy pass
        if hasattr(self.backbone, "num_features"):
            in_features = self.backbone.num_features
        else:
            # Fallback: Run a dummy forward pass to get feature shape
            # Create a dummy input with the configured image size
            dummy_input = torch.randn(1, 3, Config.img_size, Config.img_size)
            with torch.no_grad():
                features = self.backbone(dummy_input)
            in_features = features.shape[1]

        # Define the custom head
        self.pooling = GeM()
        self.fc = nn.Linear(in_features, Config.num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        # Extract features from backbone
        # Shape: (B, C, H, W)
        features = self.backbone(x)

        # Apply Generalized Mean Pooling
        # Shape: (B, C, 1, 1)
        pooled_features = self.pooling(features)

        # Flatten spatial dimensions
        # Shape: (B, C)
        flattened_features = torch.flatten(pooled_features, 1)

        # Classification
        # Shape: (B, num_classes)
        logits = self.fc(flattened_features)

        return logits
