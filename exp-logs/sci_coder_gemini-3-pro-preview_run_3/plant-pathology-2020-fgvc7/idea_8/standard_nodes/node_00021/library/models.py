import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the generalized mean of each channel in the feature map.
    Formula: f = (1/|X| * sum(x^p))^(1/p)
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter initialized to 3.0
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Apply clamping to avoid numerical instability with pow
        # AvgPool2d over the spatial dimensions (H, W) effectively computes the mean
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


class AppleEfficientNet(nn.Module):
    """
    EfficientNet-B4 based model with Multi-Level GeM Pooling.
    Extracts features from the last 3 stages (strides 8, 16, 32).
    """

    def __init__(self, model_name=Config.MODEL_EFFNET, pretrained=True):
        super().__init__()

        # Load backbone with features_only=True to get intermediate feature maps
        # out_indices=(2, 3, 4) typically corresponds to strides 8, 16, 32 for EfficientNet
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, features_only=True, out_indices=(2, 3, 4)
        )

        # Dynamic feature dimension calculation via dummy forward pass (Cite debug_lesson_6)
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.backbone(dummy_input)

        self.in_features = [f.shape[1] for f in features]

        # Create a GeM pooling layer for each extracted feature level
        self.gem_layers = nn.ModuleList([GeM() for _ in range(len(self.in_features))])

        # Calculate total input dimension for the linear head after concatenation
        total_features = sum(self.in_features)

        # Linear classification head
        self.fc = nn.Linear(total_features, Config.NUM_CLASSES)

    def forward(self, x):
        # Backbone returns a list of tensors corresponding to out_indices
        features = self.backbone(x)

        pooled_features = []
        for i, feat in enumerate(features):
            # Apply GeM pooling: (B, C, H, W) -> (B, C, 1, 1)
            pooled = self.gem_layers[i](feat)
            # Flatten: (B, C, 1, 1) -> (B, C)
            pooled = pooled.flatten(1)
            pooled_features.append(pooled)

        # Concatenate features from all levels: (B, C1 + C2 + C3)
        concat_features = torch.cat(pooled_features, dim=1)

        # Classification
        output = self.fc(concat_features)
        return output


class AppleMaxViT(nn.Module):
    """
    MaxViT-Tiny based model with Multi-Level GeM Pooling.
    Extracts features from the last 3 stages.
    """

    def __init__(self, model_name=Config.MODEL_MAXVIT, pretrained=True):
        super().__init__()

        # Load backbone with features_only=True
        # out_indices=(2, 3, 4) corresponds to the hierarchical stages
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, features_only=True, out_indices=(2, 3, 4)
        )

        # Dynamic feature dimension calculation via dummy forward pass (Cite debug_lesson_6)
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.backbone(dummy_input)

        self.in_features = [f.shape[1] for f in features]

        # GeM pooling layers
        self.gem_layers = nn.ModuleList([GeM() for _ in range(len(self.in_features))])

        # Classification Head
        total_features = sum(self.in_features)
        self.fc = nn.Linear(total_features, Config.NUM_CLASSES)

    def forward(self, x):
        # Backbone returns a list of tensors
        features = self.backbone(x)

        pooled_features = []
        for i, feat in enumerate(features):
            # Apply GeM pooling
            pooled = self.gem_layers[i](feat)
            # Flatten
            pooled = pooled.flatten(1)
            pooled_features.append(pooled)

        # Concatenate
        concat_features = torch.cat(pooled_features, dim=1)

        # Classification
        output = self.fc(concat_features)
        return output
