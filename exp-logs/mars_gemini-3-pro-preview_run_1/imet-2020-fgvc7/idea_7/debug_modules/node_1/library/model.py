import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.

    Computes the generalized mean of the spatial features.
    f(X) = (1/|X| * sum(x^p))^(1/p)

    When p=1, it approximates Average Pooling.
    When p -> infinity, it approximates Max Pooling.
    The parameter p is learnable.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter, initialized to 3.0
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Clamp to avoid NaN gradients for negative values (though ConvNeXt output is usually ReLU'd/GELU'd)
        # or zeros when raising to power < 1.
        # ConvNeXt output features can be negative depending on the stage/activation,
        # but usually GeM is applied on ReLU outputs or we clamp.
        # We clamp min=eps to ensure stability.
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
    Artwork Attribute Labeling Model.

    Architecture:
    1. Backbone: ConvNeXt-Small (Pretrained)
    2. Pooling: GeM (Generalized Mean Pooling)
    3. Head: Linear Layer (num_features -> num_classes)
    """

    def __init__(self, model_name=Config.MODEL_NAME, pretrained=Config.PRETRAINED):
        super(ArtworkModel, self).__init__()

        # Load backbone from timm
        # num_classes=0 and global_pool="" ensures we get the spatial feature map
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Get the number of input features for the head
        # For convnext_small, this is typically 768
        in_features = self.backbone.num_features

        # Pooling layer
        self.pooling = GeM()

        # Classification head
        self.head = nn.Linear(in_features, Config.NUM_CLASSES)

    def forward(self, x):
        # Extract features: (Batch, Channels, Height, Width)
        features = self.backbone(x)

        # Apply GeM pooling: (Batch, Channels, 1, 1)
        pooled = self.pooling(features)

        # Flatten: (Batch, Channels)
        flatten = torch.flatten(pooled, 1)

        # Classification logits: (Batch, NumClasses)
        logits = self.head(flatten)

        return logits
