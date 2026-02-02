import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library import config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        # Clamp to avoid NaN gradients with power
        x = torch.clamp(x, min=self.eps)

        # Apply GeM formula: (AvgPool(x^p))^(1/p)
        x_pow = torch.pow(x, self.p)

        # Global Average Pooling on the powered tensor
        # Kernel size is the spatial dimensions of the feature map
        pooled = F.avg_pool2d(x_pow, (x.size(-2), x.size(-1)))

        # Root
        x_out = torch.pow(pooled, 1.0 / self.p)
        return x_out


class WhaleModel(nn.Module):
    """
    Whale Detection Model wrapping a timm backbone with GeM pooling and a custom head.
    """

    def __init__(self, model_name, pretrained=True):
        super(WhaleModel, self).__init__()

        # Load backbone from timm
        # in_chans=1: Adapts the first layer for 1-channel input (spectrograms)
        # num_classes=0: Removes the default fully connected layer
        # global_pool="": Removes the default pooling, returning spatial features (N, C, H, W)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, in_chans=1, num_classes=0, global_pool=""
        )

        # Determine the number of output channels from the backbone
        # We use a dummy forward pass to be robust across different architectures
        with torch.no_grad():
            # Create a dummy input with shape (1, 1, 128, 128)
            # Size 128x128 is arbitrary, just to get channel depth
            dummy_input = torch.randn(1, 1, 128, 128)
            features = self.backbone(dummy_input)
            in_features = features.shape[1]

        # Custom Head
        if config.USE_GEM_POOLING:
            self.pooling = GeM()
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)

        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(in_features, config.NUM_CLASSES)

    def forward(self, x):
        # x shape: (Batch, 1, F, T)

        # Extract features
        features = self.backbone(x)  # -> (Batch, C, H, W)

        # Pooling
        pooled = self.pooling(features)  # -> (Batch, C, 1, 1)

        # Flatten
        flattened = pooled.view(pooled.size(0), -1)  # -> (Batch, C)

        # Dropout and Classifier
        dropped = self.drop(flattened)
        logits = self.fc(dropped)  # -> (Batch, 1)

        return logits


def get_model(model_name, pretrained=True):
    """
    Factory function to create the model.
    """
    return WhaleModel(model_name, pretrained=pretrained)
