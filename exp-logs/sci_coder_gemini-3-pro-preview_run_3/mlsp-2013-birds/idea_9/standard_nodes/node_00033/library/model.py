import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class BirdResNet18(nn.Module):
    """
    Standard ResNet-18 for Bird Species Classification.
    Cite {solution_lesson_node_00030}: Global Resizing vs. Multi-Instance Tiling
    """

    def __init__(self):
        super(BirdResNet18, self).__init__()

        # Load ResNet-18 backbone
        if Config.PRETRAINED:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.backbone = models.resnet18(weights=weights)

        # Replace the final fully connected layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Height, Width)

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes)
        """
        return self.backbone(x)
