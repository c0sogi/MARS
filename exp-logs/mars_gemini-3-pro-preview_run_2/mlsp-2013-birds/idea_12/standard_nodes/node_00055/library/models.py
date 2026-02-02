import torch
import torch.nn as nn
import timm
from library.config import Config


class BirdClassifier(nn.Module):
    """
    Bird Species Classifier wrapper around timm backbones.

    This class instantiates a CNN backbone (e.g., ResNet18, DenseNet121),
    applies Global Average Pooling (GAP), and projects to the target class logits.
    It handles the heterogeneous input resolutions defined in the ensemble strategy
    via the GAP layer.
    """

    def __init__(self, model_name, pretrained=True):
        """
        Args:
            model_name (str): The name of the timm model to create (e.g., 'resnet18', 'densenet121').
            pretrained (bool): Whether to load ImageNet pretrained weights.
        """
        super(BirdClassifier, self).__init__()

        # Create the model using timm
        # - pretrained=True: Loads weights from ImageNet
        # - num_classes=Config.NUM_CLASSES: Replaces the head with a Linear layer for 19 classes
        # - in_chans=3: Configures the first layer to accept Pseudo-RGB images
        # - global_pool='avg': Ensures the architecture uses Global Average Pooling before the head,
        #   which allows the model to handle variable input sizes (e.g., 224x448 vs 160x320).
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=Config.NUM_CLASSES,
            in_chans=3,
            global_pool="avg",
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 3, Height, Width).

        Returns:
            torch.Tensor: Raw logits of shape (Batch, Num_Classes).
                          Sigmoid activation is applied later (in loss or inference).
        """
        return self.model(x)


def get_model(model_name, pretrained=True, device=None):
    """
    Factory function to create and configure the model.

    Args:
        model_name (str): Name of the architecture.
        pretrained (bool): Whether to use pretrained weights.
        device (torch.device, optional): Device to move the model to.

    Returns:
        BirdClassifier: The initialized model.
    """
    model = BirdClassifier(model_name, pretrained=pretrained)

    if device:
        model.to(device)

    return model
