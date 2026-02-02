import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes f(x) = (mean(x^p))^(1/p).

    Args:
        p (float): Initial value for the power parameter.
        eps (float): Small constant for numerical stability.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Clamp min value to eps to avoid NaN when taking power of negative numbers or zero
        # Note: Input to GeM in this architecture is ReLU'd, so it's non-negative.
        # We clamp to eps to handle the zero case for the power operation stability.
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


class DenseNet121GeM(nn.Module):
    """
    DenseNet121 with modified stem and GeM pooling for pathology classification.

    Modifications:
    1. Input Stem: Replaces 7x7 stride-2 conv and 3x3 max pool with a single 3x3 stride-1 conv.
       This preserves spatial resolution for small 48x48 inputs.
    2. Pooling: Replaces Global Average Pooling with GeM Pooling.
    """

    def __init__(self, pretrained=True):
        super(DenseNet121GeM, self).__init__()

        # Load Pretrained Backbone
        weights = "IMAGENET1K_V1" if pretrained else None
        self.backbone = models.densenet121(weights=weights)

        # 1. Modify Stem
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # We replace it with 3x3 stride 1 to prevent initial downsampling
        self.backbone.features.conv0 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # Remove the initial MaxPool2d(kernel_size=3, stride=2, padding=1)
        # This further preserves spatial dimensions.
        self.backbone.features.pool0 = nn.Identity()

        # 2. Setup Classifier Head
        in_features = self.backbone.classifier.in_features

        # Remove original classifier (we will implement our own head)
        self.backbone.classifier = nn.Identity()

        # GeM Pooling Layer
        # Initialized with p from Config (default 3.0)
        self.gem = GeM(p=Config.GEM_P_INIT)

        # Final Linear Classifier
        self.fc = nn.Linear(in_features, Config.NUM_CLASSES)

    def forward(self, x):
        # Extract features using the modified backbone
        # Output shape: (Batch, 1024, H_feat, W_feat)
        features = self.backbone.features(x)

        # DenseNet specific: The 'features' sequential ends with BatchNorm.
        # The original forward pass applies ReLU before pooling.
        # We must apply ReLU here manually before GeM.
        features = F.relu(features, inplace=True)

        # Apply Generalized Mean Pooling
        # Output shape: (Batch, 1024, 1, 1)
        pooled = self.gem(features)

        # Flatten
        # Output shape: (Batch, 1024)
        flattened = torch.flatten(pooled, 1)

        # Classification
        # Output shape: (Batch, 1)
        logits = self.fc(flattened)

        return logits
