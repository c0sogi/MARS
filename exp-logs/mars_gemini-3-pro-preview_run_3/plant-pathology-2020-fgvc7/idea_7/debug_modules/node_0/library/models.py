import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Learns a parameter p to interpolate between Average Pooling and Max Pooling.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x: (B, C, H, W)
        # Clamp min value to avoid NaN in power
        x = x.clamp(min=eps)
        # Average pooling on x^p
        # Output size becomes (B, C, 1, 1)
        x_p = x.pow(p)
        avg_pool = F.avg_pool2d(x_p, (x.size(-2), x.size(-1)))
        # Raise to power 1/p
        return avg_pool.pow(1.0 / p)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class AppleEfficientNet(nn.Module):
    """
    EfficientNet-B4 backbone with Multi-Level GeM Pooling.
    Extracts features from strides 8, 16, and 32.
    """

    def __init__(self, model_name=Config.EFFNET_MODEL_NAME, pretrained=True):
        super(AppleEfficientNet, self).__init__()

        # Load backbone with features_only=True
        # indices (2, 3, 4) correspond to strides 8, 16, 32 for EfficientNet
        self.backbone = timm.create_model(
            model_name, features_only=True, out_indices=(2, 3, 4), pretrained=pretrained
        )

        # Get channel counts for the selected indices
        feature_channels = self.backbone.feature_info.channels()

        # Create a GeM pooling layer for each feature level
        self.gem_pools = nn.ModuleList([GeM() for _ in range(len(feature_channels))])

        # Calculate total input dimension for the classifier
        total_features = sum(feature_channels)

        # Final Classifier
        self.fc = nn.Linear(total_features, Config.NUM_CLASSES)

    def forward(self, x):
        # Extract features
        features = self.backbone(x)

        pooled_features = []
        for i, feat in enumerate(features):
            # Apply GeM pooling
            pooled = self.gem_pools[i](feat)
            # Flatten: (B, C, 1, 1) -> (B, C)
            pooled = pooled.flatten(1)
            pooled_features.append(pooled)

        # Concatenate features from all levels
        concat_features = torch.cat(pooled_features, dim=1)

        # Classification
        output = self.fc(concat_features)
        return output


class AppleSwin(nn.Module):
    """
    Swin Transformer Small backbone with Multi-Stage GeM Pooling.
    Extracts features from Stages 2, 3, and 4.
    """

    def __init__(self, model_name=Config.SWIN_MODEL_NAME, pretrained=True):
        super(AppleSwin, self).__init__()

        # Load backbone with features_only=True
        # indices (1, 2, 3) correspond to Stage 2, 3, 4 (strides 8, 16, 32)
        self.backbone = timm.create_model(
            model_name, features_only=True, out_indices=(1, 2, 3), pretrained=pretrained
        )

        # Get channel counts
        feature_channels = self.backbone.feature_info.channels()

        # Create GeM pooling layers
        self.gem_pools = nn.ModuleList([GeM() for _ in range(len(feature_channels))])

        # Calculate total input dimension
        total_features = sum(feature_channels)

        # Final Classifier
        self.fc = nn.Linear(total_features, Config.NUM_CLASSES)

    def forward(self, x):
        # Extract features
        # timm ensures Swin features are returned in NCHW format when features_only=True
        features = self.backbone(x)

        pooled_features = []
        for i, feat in enumerate(features):
            # Apply GeM pooling
            pooled = self.gem_pools[i](feat)
            # Flatten
            pooled = pooled.flatten(1)
            pooled_features.append(pooled)

        # Concatenate
        concat_features = torch.cat(pooled_features, dim=1)

        # Classification
        output = self.fc(concat_features)
        return output
