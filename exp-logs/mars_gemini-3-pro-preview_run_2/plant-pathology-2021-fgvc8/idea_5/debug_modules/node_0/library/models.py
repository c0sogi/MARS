import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the generalized mean of each channel in the feature map.
    f(X) = (1/|X| * sum(x^p))^(1/p)
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x is expected to be (B, C, H, W)
        # Clamp to avoid NaN with power, then avg pool, then root p
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


class AppleDiseaseModel(nn.Module):
    """
    Unified model class for Apple Disease Detection.
    Supports both ConvNeXt and Swin Transformer backbones via timm.
    Replaces default pooling with GeM and adds a linear classification head.
    """

    def __init__(self, model_name, pretrained=True):
        super(AppleDiseaseModel, self).__init__()

        # Load backbone with no classification head and no global pooling
        # This returns the raw feature maps
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Determine input features for the head
        self.in_features = self.backbone.num_features

        # Learnable Pooling layer
        self.pooling = GeM()

        # Classification Head
        self.fc = nn.Linear(self.in_features, Config.NUM_CLASSES)

    def forward(self, x):
        # Extract features
        features = self.backbone.forward_features(x)

        # Handle different output shapes from timm backbones
        # ConvNeXt typically returns: (B, C, H, W)
        # Swin Transformers typically return: (B, H, W, C)

        if features.ndim == 4:
            # Check if channels are last (Swin case)
            # We assume C matches self.in_features
            if features.shape[-1] == self.in_features:
                features = features.permute(0, 3, 1, 2)

        # Apply GeM pooling -> (B, C, 1, 1)
        pooled_features = self.pooling(features)

        # Flatten: (B, C, 1, 1) -> (B, C)
        flattened_features = pooled_features.view(pooled_features.size(0), -1)

        # Classification
        logits = self.fc(flattened_features)

        return logits
