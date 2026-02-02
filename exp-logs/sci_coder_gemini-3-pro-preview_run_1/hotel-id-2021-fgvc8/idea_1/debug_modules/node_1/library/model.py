import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class HotelResNet(nn.Module):
    """
    ResNet-18 based model for Hotel Identification.

    This class implements a standard ResNet-18 backbone where the final
    fully connected layer is replaced to output logits for the specific
    number of hotel classes in the dataset.
    """

    def __init__(self, n_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED):
        """
        Initialize the HotelResNet model.

        Args:
            n_classes (int): The number of target classes (unique hotels).
                             Defaults to Config.NUM_CLASSES.
            pretrained (bool): Whether to use ImageNet pre-trained weights.
                               Defaults to Config.PRETRAINED.
        """
        super(HotelResNet, self).__init__()

        # Determine the weights to load
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None

        # Load the ResNet-18 backbone
        # We assign it to self.backbone to keep the structure clean,
        # though often in torchvision it's just self.model
        self.backbone = models.resnet18(weights=weights)

        # The original fc layer in ResNet-18 is:
        # (fc): Linear(in_features=512, out_features=1000, bias=True)
        # We need to replace it to output n_classes.
        in_features = self.backbone.fc.in_features

        # Replace the fully connected layer
        self.backbone.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, Channels, Height, Width).

        Returns:
            torch.Tensor: Output logits of shape (Batch_Size, n_classes).
        """
        return self.backbone(x)
