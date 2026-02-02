import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from library.config import Config


class BirdResNet(nn.Module):
    """
    Standard ResNet-34 architecture for Bird Species Classification.
    Cite solution_lesson_node_00068: Architectural Homogeneity.
    """

    def __init__(self, num_classes=19, pretrained=True):
        super(BirdResNet, self).__init__()

        # Load ResNet-34 Backbone
        try:
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet34(weights=weights)
        except AttributeError:
            backbone = models.resnet34(pretrained=pretrained)

        # Use standard backbone
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Classification Head
        self.in_features = backbone.fc.in_features
        self.fc = nn.Linear(self.in_features, num_classes)

    def forward(self, x):
        # Standard Forward Pass
        out = self.backbone(x)
        out = torch.flatten(out, 1)
        logits = self.fc(out)
        return logits
