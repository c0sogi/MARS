import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library import config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the generalized mean of the input feature map: f(X) = (avg(X^p))^(1/p).
    This allows the model to learn to focus on salient features (peaks) rather than just averaging,
    which is beneficial for detecting transient signals like whale calls.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: (N, C, H, W)
        # We pool over the spatial dimensions (H, W)
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)


class WhaleEfficientNet(nn.Module):
    """
    EfficientNet-B0 adapted for 1-channel audio spectrograms with GeM pooling.
    """

    def __init__(self, pretrained=True):
        super(WhaleEfficientNet, self).__init__()

        # Load the pretrained EfficientNet-B0 backbone
        # Using DEFAULT weights corresponds to the best available ImageNet weights
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        # 1. Adapt the first convolutional layer for 1-channel input
        # Original layer: Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        original_conv = self.backbone.features[0][0]

        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        if pretrained:
            # Initialize the 1-channel weights by averaging the original 3-channel RGB weights
            # This preserves the spatial structure of learned filters
            with torch.no_grad():
                new_conv.weight[:] = original_conv.weight.mean(dim=1, keepdim=True)

        # Replace the layer in the backbone
        self.backbone.features[0][0] = new_conv

        # 2. Replace Global Average Pooling with GeM Pooling
        if config.USE_GEM_POOLING:
            self.backbone.avgpool = GeM()

        # 3. Replace the Classification Head
        # The original classifier is a Sequential block (Dropout -> Linear)
        # We replace it with a single Linear layer for our binary classification task

        # Retrieve the input features size of the final linear layer (1280 for B0)
        if isinstance(self.backbone.classifier, nn.Sequential):
            in_features = self.backbone.classifier[-1].in_features
        else:
            in_features = self.backbone.classifier.in_features

        self.backbone.classifier = nn.Linear(in_features, config.NUM_CLASSES)

    def forward(self, x):
        # EfficientNet forward pass: features -> avgpool (GeM) -> flatten -> classifier
        return self.backbone(x)
