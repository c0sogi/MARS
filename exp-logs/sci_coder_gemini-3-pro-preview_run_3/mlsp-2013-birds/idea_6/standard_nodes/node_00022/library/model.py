import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import Config


class BirdResNetSPP(nn.Module):
    """
    Standard ResNet-18 for Multi-Label Bird Species Classification.
    Renamed class kept for compatibility with imports, but implementation is standard ResNet.
    Cite Lesson 00019: Use standard architecture on small datasets.
    """

    def __init__(
        self,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
        spp_levels=None,  # Ignored
    ):
        super(BirdResNetSPP, self).__init__()

        # 1. Load Pretrained ResNet-18 Backbone
        try:
            from torchvision.models import ResNet18_Weights

            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
        except ImportError:
            self.backbone = models.resnet18(pretrained=pretrained)

        # 2. Replace Classifier
        # ResNet18 fc input dim is 512
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.backbone(x)
