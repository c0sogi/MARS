import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes (1/N * sum(x^p))^(1/p).
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x shape: (Batch, Channels, Height, Width)
        # Pool over spatial dimensions (H, W)
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )


class WhaleEfficientNet(nn.Module):
    """
    EfficientNet-B0 (Noisy Student) modified for 1-channel input and GeM pooling.
    """

    def __init__(self, model_name=Config.MODEL_A_NAME, pretrained=Config.PRETRAINED):
        super(WhaleEfficientNet, self).__init__()

        # Load backbone with no classifier and no global pooling (return spatial features)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Modify the first convolutional layer (conv_stem) to accept 1 channel
        if hasattr(self.backbone, "conv_stem"):
            old_layer = self.backbone.conv_stem
            new_layer = nn.Conv2d(
                in_channels=Config.IN_CHANNELS,
                out_channels=old_layer.out_channels,
                kernel_size=old_layer.kernel_size,
                stride=old_layer.stride,
                padding=old_layer.padding,
                bias=old_layer.bias is not None,
            )

            # Initialize weights by averaging the original RGB weights
            with torch.no_grad():
                new_layer.weight[:] = torch.mean(old_layer.weight, dim=1, keepdim=True)
                if old_layer.bias is not None:
                    new_layer.bias[:] = old_layer.bias

            self.backbone.conv_stem = new_layer
        else:
            raise AttributeError(f"Backbone {model_name} does not have 'conv_stem'.")

        # Pooling and Classifier
        self.pooling = GeM() if Config.USE_GEM_POOLING else nn.AdaptiveAvgPool2d(1)
        self.in_features = self.backbone.num_features
        self.fc = nn.Linear(self.in_features, Config.NUM_CLASSES)

    def forward(self, x):
        # x: (Batch, 1, Freq, Time)
        features = self.backbone(x)  # (Batch, C, H, W)
        pooled = self.pooling(features)  # (Batch, C, 1, 1)
        flattened = pooled.view(pooled.size(0), -1)  # (Batch, C)
        output = self.fc(flattened)  # (Batch, 1)
        return output
