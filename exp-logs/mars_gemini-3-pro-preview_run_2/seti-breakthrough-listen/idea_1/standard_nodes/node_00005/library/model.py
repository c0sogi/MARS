import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class TechnosignatureResNet(nn.Module):
    """
    ResNet18 backbone adapted for Technosignature Detection.
    Cite solution_lesson_node_00004: Using a deeper backbone with vertically stacked inputs.
    """

    def __init__(self):
        super(TechnosignatureResNet, self).__init__()

        # Load ResNet18 with ImageNet weights
        self.backbone = models.resnet18(weights="DEFAULT")

        # Modify first conv layer to accept 1 channel (instead of 3)
        old_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=Config.NUM_CHANNELS,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias,
        )

        # Initialize new conv1 weights by averaging original weights across channels
        with torch.no_grad():
            self.backbone.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

        # Modify Output Layer for Binary Classification
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.backbone(x)
