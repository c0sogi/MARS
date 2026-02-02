import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class HotelResNet(nn.Module):
    """
    A ResNet-18 based neural network for Hotel Identification.

    This model utilizes a ResNet-18 backbone, optionally pre-trained on ImageNet.
    The final fully connected layer is replaced to output logits for the specific
    number of hotel classes in the dataset.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED):
        """
        Initialize the HotelResNet model.

        Args:
            num_classes (int): The number of target hotel classes.
                               Defaults to Config.NUM_CLASSES.
            pretrained (bool): Whether to initialize with pre-trained ImageNet weights.
                               Defaults to Config.PRETRAINED.
        """
        super(HotelResNet, self).__init__()

        # Select weights based on the pretrained flag
        # 'DEFAULT' corresponds to the best available weights (IMAGENET1K_V1 for ResNet18)
        weights = "DEFAULT" if pretrained else None

        # Load the ResNet-18 backbone
        self.backbone = models.resnet18(weights=weights)

        # The standard ResNet-18 architecture ends with:
        #   (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
        #   (fc): Linear(in_features=512, out_features=1000, bias=True)

        # We replace the 'fc' layer (Head) to map to our specific number of classes.
        # First, retrieve the number of input features for the fc layer (512 for ResNet18)
        in_features = self.backbone.fc.in_features

        # Replace the layer
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input batch of images with shape (Batch, Channels, Height, Width).

        Returns:
            torch.Tensor: Raw logits with shape (Batch, num_classes).
        """
        # The torchvision ResNet implementation handles the full forward pass,
        # including the conv layers, average pooling, flattening, and the (modified) fc layer.
        return self.backbone(x)
