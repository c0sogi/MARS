import torch
import torch.nn as nn
import timm
from library.config import Config


class CassavaClassifier(nn.Module):
    """
    Cassava Leaf Disease Classifier using timm models.

    This class wraps timm's model creation to provide a consistent interface
    for different backbone architectures (e.g., ViT, BEiT) used in the ensemble.
    It handles the replacement of the classification head to match the specific
    number of classes in the Cassava dataset.
    """

    def __init__(self, model_name, num_classes=Config.num_classes, pretrained=True):
        """
        Initialize the model.

        Args:
            model_name (str): The name of the model architecture to load from timm.
                              (e.g., 'vit_base_patch16_384', 'beit_base_patch16_384')
            num_classes (int): The number of classes for the classification head.
                               Defaults to Config.num_classes (5).
            pretrained (bool): Whether to load pretrained weights (usually ImageNet).
                               Defaults to True.
        """
        super(CassavaClassifier, self).__init__()

        # Create the model using timm.
        # timm.create_model handles:
        # 1. Loading the architecture.
        # 2. Loading pretrained weights if pretrained=True.
        # 3. Replacing the original classification head with a new one
        #    matching num_classes.
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=Config.drop_path_rate,
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input batch of images [B, C, H, W].

        Returns:
            torch.Tensor: Raw logits [B, num_classes].
        """
        return self.model(x)
