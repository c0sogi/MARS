import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes the generalized mean of each channel in the input feature map.
    Formula: f(X) = (1/|X| * sum(x^p))^(1/p)
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter initialized to 3.0
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Clamp input to eps to avoid numerical instability with power
        # Average pooling of x^p
        x = x.clamp(min=eps)
        return F.avg_pool2d(x.pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

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
    Artwork Attribute Labeling Model.
    Architecture:
    1. Backbone: ConvNeXt-Small (pretrained on ImageNet-22k, finetuned on 1k)
    2. Pooling: Generalized Mean Pooling (GeM) or Adaptive Avg Pooling
    3. Head: LayerNorm -> Linear
    """

    def __init__(self, pretrained=True):
        super(ArtworkModel, self).__init__()

        # Load backbone using timm
        # num_classes=0 and global_pool="" removes the default head and pooling
        # This returns the feature map (B, C, H, W)
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        # Get the number of input features for the final layer
        in_features = self.backbone.num_features

        # Pooling Layer
        if Config.USE_GEM_POOLING:
            self.pooling = GeM()
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)

        # Classification Head
        # ConvNeXt architecture typically uses LayerNorm before the final classifier
        self.norm = nn.LayerNorm(in_features)
        self.fc = nn.Linear(in_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images of shape (Batch, 3, H, W)

        Returns:
            torch.Tensor: Logits of shape (Batch, NUM_CLASSES)
        """
        # Extract features: (B, C, H, W)
        features = self.backbone(x)

        # Pooling: (B, C, 1, 1)
        pooled_features = self.pooling(features)

        # Flatten: (B, C)
        flattened_features = pooled_features.view(pooled_features.size(0), -1)

        # Normalization
        norm_features = self.norm(flattened_features)

        # Classification: (B, NUM_CLASSES)
        logits = self.fc(norm_features)

        return logits
