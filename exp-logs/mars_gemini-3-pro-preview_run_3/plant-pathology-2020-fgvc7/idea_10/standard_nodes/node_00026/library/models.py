import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the generalized mean of the input tensor.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x: (Batch, Channels, Height, Width)
        # Clamp min to eps to avoid numerical instability with pow
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )


class HeterogeneousExpert(nn.Module):
    """
    A wrapper class for Heterogeneous Expert models (EfficientNetV2, MaxViT, etc.).
    Extracts features from the last three stages, applies GeM pooling, and concatenates them.
    """

    def __init__(self, backbone_name, num_classes, pretrained=True):
        """
        Args:
            backbone_name (str): Name of the timm backbone to use.
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to load pretrained weights.
        """
        super(HeterogeneousExpert, self).__init__()

        # Initialize backbone with features_only=True to get intermediate feature maps.
        # We extract the last 3 stages (indices 2, 3, 4) to capture multi-scale information.
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 3, 4),
        )

        # Get the number of channels for each extracted feature map
        feature_info = self.backbone.feature_info.channels()

        # Create a GeM pooling layer for each extracted feature map
        self.gem_pools = nn.ModuleList([GeM() for _ in range(len(feature_info))])

        # Calculate total input features for the final linear layer
        # We concatenate the pooled features, so we sum the channel counts
        total_features = sum(feature_info)

        # Classification Head
        self.fc = nn.Linear(total_features, num_classes)

    def forward(self, x):
        # Forward pass through backbone to get list of feature maps
        features = self.backbone(x)

        # Apply GeM pooling to each feature map and flatten
        # features[i] shape: (B, C_i, H_i, W_i) -> GeM -> (B, C_i, 1, 1) -> Flatten -> (B, C_i)
        pooled_features = [
            gem(f).flatten(1) for gem, f in zip(self.gem_pools, features)
        ]

        # Concatenate the pooled features along the channel dimension
        # Result shape: (B, Sum(C_i))
        concat_features = torch.cat(pooled_features, dim=1)

        # Final classification
        output = self.fc(concat_features)

        return output
