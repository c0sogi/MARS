import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import CFG


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes the generalized mean of the input tensor, which is a learnable
    pooling operation that generalizes Max Pooling (p -> infinity) and
    Average Pooling (p = 1).
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x is expected to be (B, C, H, W)
        # Clamp min value to avoid NaN gradients with pow
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
    Apple Disease Detection Model.
    Uses a backbone from timm (ConvNeXt or Swin V2), replaces the global pooling
    with GeM, and adds a linear classification head.
    """

    def __init__(self, model_name, pretrained=True):
        super(AppleDiseaseModel, self).__init__()

        # Load backbone with no classification head and no global pooling
        # This returns the raw feature maps
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Determine the number of input features for the head
        self.in_features = self.backbone.num_features

        # Custom Pooling Layer
        self.pooling = GeM()

        # Classification Head
        # Output dimension is determined by CFG.num_classes (6)
        self.fc = nn.Linear(self.in_features, CFG.num_classes)

    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)

        # Handle dimension ordering differences between CNNs and Transformers
        # ConvNeXt output: (B, C, H, W)
        # Swin Transformer output: (B, H, W, C)
        if features.dim() == 4:
            # Check if channels are in the last dimension
            if features.shape[-1] == self.in_features:
                # Permute to (B, C, H, W) for GeM pooling
                features = features.permute(0, 3, 1, 2)

        # Apply GeM Pooling -> (B, C, 1, 1)
        pooled_features = self.pooling(features)

        # Flatten -> (B, C)
        flattened_features = pooled_features.flatten(1)

        # Classification -> (B, num_classes)
        logits = self.fc(flattened_features)

        return logits
