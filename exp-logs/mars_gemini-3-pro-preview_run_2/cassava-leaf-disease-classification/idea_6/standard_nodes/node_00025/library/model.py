import torch
import torch.nn as nn
import timm
from library.config import CFG


class CassavaClassifier(nn.Module):
    """
    Cassava Leaf Disease Classifier based on the ConvNeXt architecture.

    This class wraps a timm model, configuring it with the specific
    architecture, pretrained weights, class count, and regularization
    parameters defined in the configuration.
    """

    def __init__(self, model_name=CFG.model_name, pretrained=True):
        """
        Initialize the CassavaClassifier.

        Args:
            model_name (str): The name of the timm model to load.
                              Defaults to CFG.model_name.
            pretrained (bool): Whether to load pretrained weights.
                               Defaults to True.
        """
        super(CassavaClassifier, self).__init__()

        # Initialize the backbone using timm.
        # - num_classes=CFG.num_classes (5) automatically replaces the original
        #   ImageNet head with a new linear layer for our specific task.
        # - drop_path_rate=CFG.drop_path_rate (0.4) sets the stochastic depth
        #   regularization intensity.
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=CFG.num_classes,
            drop_path_rate=CFG.drop_path_rate,
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input batch of images.

        Returns:
            torch.Tensor: Raw logits for the 5 classes.
        """
        return self.model(x)
