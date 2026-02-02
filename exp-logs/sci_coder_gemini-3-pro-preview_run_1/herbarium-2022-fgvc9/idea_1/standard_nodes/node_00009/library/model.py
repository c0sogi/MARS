import torch
import torch.nn as nn
import torchvision
from library import config


class LinearProbeResNet(nn.Module):
    """
    A ResNet-18 based model for linear probing.
    The backbone is initialized with ImageNet weights and frozen.
    The classification head is replaced to match the specific number of plant classes.
    """

    def __init__(
        self, num_classes=config.NUM_CLASSES, freeze_backbone=config.FREEZE_BACKBONE
    ):
        """
        Args:
            num_classes (int): Number of output classes.
            freeze_backbone (bool): Whether to freeze the convolutional backbone.
        """
        super(LinearProbeResNet, self).__init__()

        # Load pre-trained ResNet-50 with ImageNet V2 weights
        # Using the modern 'weights' parameter for torchvision >= 0.13
        # Cite solution_lesson_node_00006: Capacity and Initialization Quality
        self.backbone = torchvision.models.resnet50(weights="IMAGENET1K_V2")

        # Freeze backbone parameters if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the final fully connected layer
        # The original fc layer in ResNet-18 is: (fc): Linear(in_features=512, out_features=1000, bias=True)
        in_features = self.backbone.fc.in_features

        # The new layer will automatically have requires_grad=True
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input batch of images [B, C, H, W].

        Returns:
            torch.Tensor: Raw logits [B, num_classes].
        """
        return self.backbone(x)


def get_model(num_classes=config.NUM_CLASSES, freeze_backbone=config.FREEZE_BACKBONE):
    """
    Factory function to create an instance of LinearProbeResNet.

    Args:
        num_classes (int): Number of target classes.
        freeze_backbone (bool): Whether to freeze the backbone.

    Returns:
        LinearProbeResNet: The configured model instance.
    """
    model = LinearProbeResNet(num_classes=num_classes, freeze_backbone=freeze_backbone)
    return model
