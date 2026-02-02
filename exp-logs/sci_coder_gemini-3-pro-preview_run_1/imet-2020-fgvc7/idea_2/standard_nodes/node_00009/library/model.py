import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.

    Computes the generalized mean of the spatial dimensions of the input tensor.
    Formula: f(X) = (1/|X| * sum(x^p))^(1/p)

    This pooling strategy is learnable (via parameter p) and interpolates between
    Average Pooling (p=1) and Max Pooling (p -> infinity). It is particularly
    effective for retrieval and fine-grained recognition tasks where focusing on
    salient regions is crucial.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3.0, eps=1e-6):
        # Clamp input to avoid numerical instability with power operations
        # Apply average pooling on x^p, then take the (1/p)-th root
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class ArtworkModel(nn.Module):
    """
    Artwork Attribute Labeling Model.

    Architecture:
    1. Backbone: ConvNeXt-Tiny (pre-trained on ImageNet).
    2. Pooling: Generalized Mean Pooling (GeM).
    3. Head: Linear layer for multi-label classification.
    """

    def __init__(self, pretrained=True):
        super(ArtworkModel, self).__init__()

        # Load the backbone from timm
        # num_classes=0 and global_pool='' ensures we get the raw feature map (B, C, H, W)
        self.backbone = timm.create_model(
            Config.model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Determine the number of input features for the head
        # ConvNeXt-Tiny typically has 768 features at the final stage
        self.in_features = self.backbone.num_features

        # Generalized Mean Pooling
        self.pooling = GeM(p=Config.gem_p)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Classification Head
        self.fc = nn.Linear(self.in_features, Config.num_classes)

    def forward(self, x):
        # Extract features: (B, C, H, W)
        features = self.backbone(x)

        # Apply GeM pooling: (B, C, 1, 1)
        pooled_features = self.pooling(features)

        # Flatten: (B, C)
        flattened_features = self.flatten(pooled_features)

        # Classification logits: (B, NumClasses)
        logits = self.fc(flattened_features)

        return logits
