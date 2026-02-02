import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.

    Computes the generalized mean: f(X) = (1/|X| * sum(x^p))^(1/p)
    This pooling method is learnable (via parameter p) and is effective at capturing
    sparse, high-activation anomalies (like lesions) without washing them out,
    unlike standard Average Pooling.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # Input x is expected to be (B, C, H, W)
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Clamp input to avoid NaN gradients with pow
        x = x.clamp(min=eps)

        # Calculate average of x^p over spatial dimensions
        # F.avg_pool2d computes 1/(H*W) * sum(x^p)
        x_pow = x.pow(p)
        avg = F.avg_pool2d(x_pow, (x.size(-2), x.size(-1)))

        # Take the p-th root
        return avg.pow(1.0 / p)


class RetinopathyModel(nn.Module):
    """
    Retinopathy Model wrapping a timm backbone with GeM pooling and a regression head.

    This class handles the integration of both CNN (EfficientNet) and Transformer (Swin)
    backbones, normalizing their output shapes for the pooling layer.
    """

    def __init__(self, model_name, pretrained=True):
        super(RetinopathyModel, self).__init__()

        # Create the backbone model using timm
        # num_classes=0: Removes the default classification head
        # global_pool='': Returns the unpooled spatial feature maps
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Determine the number of input features for the head
        self.in_features = self.backbone.num_features

        # Generalized Mean Pooling layer
        self.pooling = GeM()

        # Linear regression head (output dim = 1)
        self.head = nn.Linear(self.in_features, 1)

    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)

        # Handle different output formats from timm backbones to ensure (B, C, H, W)
        # CNNs (like EfficientNet) typically output (B, C, H, W)
        # Transformers (like Swin) typically output (B, H, W, C) in recent timm versions
        if features.ndim == 4:
            # Check if channels are in the last dimension
            if features.shape[-1] == self.in_features:
                features = features.permute(0, 3, 1, 2)

        # Apply GeM Pooling -> Output shape (B, C, 1, 1)
        pooled = self.pooling(features)

        # Flatten -> Output shape (B, C)
        flattened = pooled.flatten(1)

        # Predict continuous score -> Output shape (B, 1)
        output = self.head(flattened)

        return output
