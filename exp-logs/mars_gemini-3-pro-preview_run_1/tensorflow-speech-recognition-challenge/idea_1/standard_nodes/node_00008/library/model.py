import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class SpectroCNN(nn.Module):
    """
    Audio Classification Model using ResNet-18 Backbone.
    Cite solution_lesson_node_00004
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, input_channels=1):
        """
        Args:
            num_classes (int): Number of output classes.
            input_channels (int): Number of input channels.
        """
        super(SpectroCNN, self).__init__()

        # Use ResNet18 backbone
        # We start from scratch because ImageNet weights are for RGB images
        # and spectrogram patterns differ from natural images.
        self.backbone = models.resnet18(weights=None)

        # Modify first layer for 1 channel input (instead of 3)
        self.backbone.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Modify final layer for our number of classes
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.backbone(x)
