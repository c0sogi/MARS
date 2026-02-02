import torch
import torch.nn as nn
import timm
from library.config import CFG


class CassavaViT(nn.Module):
    """
    Vision Transformer model for Cassava Leaf Disease Classification.
    Wraps a timm model, ensuring the correct head is initialized for the specific number of classes.
    """

    def __init__(
        self,
        model_name=CFG.model_name,
        pretrained=CFG.pretrained,
        num_classes=CFG.num_classes,
    ):
        """
        Args:
            model_name (str): Name of the model architecture in timm.
            pretrained (bool): Whether to load pretrained ImageNet weights.
            num_classes (int): Number of output classes.
        """
        super(CassavaViT, self).__init__()

        # Create the model using timm
        # When num_classes is provided and differs from the pretrained model's default,
        # timm automatically resets the classifier head to a new Linear layer with the correct output size.
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input batch of images.

        Returns:
            torch.Tensor: Logits for each class.
        """
        return self.model(x)
