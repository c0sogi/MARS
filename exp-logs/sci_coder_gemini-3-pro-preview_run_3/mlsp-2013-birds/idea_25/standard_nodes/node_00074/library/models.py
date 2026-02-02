import torch
import torch.nn as nn
import timm
from library.config import Config


class BirdModel(nn.Module):
    """
    Bird Species Classification Model using heterogeneous backbones.
    Wraps timm models to provide a consistent interface and supports
    ResNet-18, EfficientNet-B0, and DenseNet-121.
    """

    def __init__(self, backbone_name: str, num_classes: int, pretrained: bool = True):
        """
        Args:
            backbone_name (str): Name of the backbone (e.g., 'resnet18', 'efficientnet_b0').
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to load ImageNet pretrained weights.
        """
        super(BirdModel, self).__init__()
        self.backbone_name = backbone_name

        # Create the model using timm
        # timm handles loading pretrained weights and replacing the classification head
        # when num_classes is specified.
        self.model = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)
