import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class BirdResNet18(nn.Module):
    """
    Standard ResNet-18 model for Bird Species Classification.
    Cite {lesson_00030}: Global Resizing vs. Multi-Instance Tiling
    """

    def __init__(self):
        super(BirdResNet18, self).__init__()

        if Config.PRETRAINED:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.backbone = models.resnet18(weights=weights)

        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 3, H, W)
        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes)
        """
        return self.backbone(x)
