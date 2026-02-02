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

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x shape: (B, C, H, W)
        # Clamp input to avoid NaN in power operation
        x = x.clamp(min=eps)
        # Apply Average Pooling to x^p
        # Output shape of avg_pool2d is (B, C, 1, 1)
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


class AppleEfficientNet(nn.Module):
    """
    EfficientNet-B4 backbone with Multi-Level GeM Pooling.
    Extracts features from strides 8, 16, and 32 (indices 2, 3, 4).
    """

    def __init__(self, pretrained=True):
        super().__init__()
        # Load backbone with feature extraction enabled
        self.backbone = timm.create_model(
            Config.EFFNET_MODEL_NAME,
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 3, 4),
        )

        # Calculate total input features for the head by summing channels from extracted stages
        feature_info = self.backbone.feature_info
        self.in_features = sum([info["num_chs"] for info in feature_info])

        self.gem = GeM(p=3)
        self.fc = nn.Linear(self.in_features, Config.NUM_CLASSES)

    def forward(self, x):
        # Backbone returns a list of feature maps
        features = self.backbone(x)

        pooled_features = []
        for f in features:
            # f: (B, C, H, W)
            # Apply GeM pooling -> (B, C, 1, 1)
            pooled = self.gem(f)
            # Flatten -> (B, C)
            pooled = pooled.flatten(1)
            pooled_features.append(pooled)

        # Concatenate features from all levels -> (B, Sum_C)
        concat_features = torch.cat(pooled_features, dim=1)

        # Classification head
        output = self.fc(concat_features)
        return output


class AppleMaxViT(nn.Module):
    """
    MaxViT-Tiny backbone with Multi-Level GeM Pooling.
    Extracts features from strides 8, 16, and 32 (indices 1, 2, 3).
    """

    def __init__(self, pretrained=True):
        super().__init__()
        # MaxViT typically has 4 stages (indices 0, 1, 2, 3) corresponding to strides 4, 8, 16, 32.
        # We select indices 1, 2, 3 to match the stride hierarchy of the EfficientNet model (8, 16, 32).
        self.backbone = timm.create_model(
            Config.MAXVIT_MODEL_NAME,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3),
        )

        feature_info = self.backbone.feature_info
        self.in_features = sum([info["num_chs"] for info in feature_info])

        self.gem = GeM(p=3)
        self.fc = nn.Linear(self.in_features, Config.NUM_CLASSES)

    def forward(self, x):
        features = self.backbone(x)

        pooled_features = []
        for f in features:
            pooled = self.gem(f)
            pooled = pooled.flatten(1)
            pooled_features.append(pooled)

        concat_features = torch.cat(pooled_features, dim=1)
        output = self.fc(concat_features)
        return output


def get_model(model_type, pretrained=True):
    """
    Factory function to initialize the appropriate model based on type.

    Args:
        model_type (str): 'effnet' or 'maxvit'
        pretrained (bool): Whether to load pretrained ImageNet weights.
    """
    if model_type == "effnet":
        return AppleEfficientNet(pretrained=pretrained)
    elif model_type == "maxvit":
        return AppleMaxViT(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
