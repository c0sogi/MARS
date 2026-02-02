import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes the generalized mean of the feature map: f = (mean(x^p))^(1/p).
    This acts as a trainable interpolation between Max Pooling (p -> infinity)
    and Average Pooling (p -> 1).
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # Clamp inputs to avoid NaN when raising negative values to float power
        # ConvNeXt outputs can be negative (GELU/LayerNorm), so clamping is essential.
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # 1. Clamp x to be strictly positive (>= eps)
        # 2. Raise to power p
        # 3. Average pool over spatial dimensions (H, W)
        # 4. Raise to power 1/p
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class ArtworkModel(nn.Module):
    """
    Deep learning model for artwork attribute classification.
    Uses a ConvNeXt-Small backbone with GeM pooling and a linear classifier.
    """

    def __init__(self, model_name=Config.MODEL_NAME, pretrained=Config.PRETRAINED):
        super().__init__()

        # Load the backbone from timm
        # global_pool='' ensures we get the (B, C, H, W) feature maps
        # num_classes=0 removes the default classification head
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Get the number of output channels from the backbone
        in_features = self.backbone.num_features

        # Initialize Pooling Layer
        if Config.USE_GEM:
            self.pooling = GeM()
        else:
            # Fallback to standard Adaptive Average Pooling if GeM is disabled
            self.pooling = nn.AdaptiveAvgPool2d(1)

        # Classification Head
        # Projects features to the number of attribute classes
        self.fc = nn.Linear(in_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W)

        Returns:
            torch.Tensor: Logits of shape (B, NUM_CLASSES)
        """
        # Extract features: (B, C, H, W)
        features = self.backbone(x)

        # Apply Pooling: (B, C, 1, 1)
        pooled_features = self.pooling(features)

        # Flatten: (B, C)
        flattened_features = pooled_features.view(pooled_features.size(0), -1)

        # Classification: (B, NUM_CLASSES)
        logits = self.fc(flattened_features)

        return logits
