import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import CFG


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes the generalized mean of the spatial features.
    f(X) = (1/|X| * sum(x^p))^(1/p)
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # Initialize p as a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

        # Check config to see if p should be learnable
        if hasattr(CFG, "gem_learnable") and not CFG.gem_learnable:
            self.p.requires_grad = False

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x: (B, C, H, W)
        # Clamp for numerical stability
        x = x.clamp(min=eps)
        # Apply GeM formula
        # AvgPool2d effectively calculates (1/N * sum(x^p))
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


class AppleClassifier(nn.Module):
    """
    Apple Disease Classifier using a timm backbone and GeM pooling.
    """

    def __init__(self, model_name, pretrained=True):
        super().__init__()

        # Load the backbone from timm
        # num_classes=0 and global_pool='' ensures we get the spatial feature map (B, C, H, W)
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Determine the number of input features for the head
        in_features = self.model.num_features

        # Initialize Pooling Layer
        if CFG.use_gem:
            self.pooling = GeM(p=CFG.gem_p)
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)

        # Classification Head
        self.fc = nn.Linear(in_features, CFG.num_classes)

    def forward(self, x):
        # Extract features from backbone
        features = self.model(x)  # Shape: (B, C, H, W)

        # Apply pooling
        pooled = self.pooling(features)  # Shape: (B, C, 1, 1)

        # Flatten
        flattened = pooled.view(pooled.size(0), -1)  # Shape: (B, C)

        # Classification
        output = self.fc(flattened)  # Shape: (B, num_classes)

        return output
