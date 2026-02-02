import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class SpectroCNN(nn.Module):
    """
    ResNet-18 based model for Audio Classification.
    Cite solution_lesson_node_00004: Using ResNet-18 backbone.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, input_channels=1):
        """
        Args:
            num_classes (int): Number of output classes.
            input_channels (int): Number of input channels.
        """
        super(SpectroCNN, self).__init__()

        # Load ResNet18 with pretrained weights
        self.backbone = models.resnet18(weights="IMAGENET1K_V1")

        # Modify first layer for 1 channel input
        # Original: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Modify fc layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.backbone(x)
