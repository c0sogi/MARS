import torch
import torch.nn as nn
import timm
from library.config import Config


class PetModel(nn.Module):
    """
    A wrapper class for timm models tailored for the Dog vs Cat classification task.

    This class instantiates a backbone architecture specified by `model_name`
    (e.g., ConvNeXt, Swin Transformer, EfficientNetV2) and modifies the head
    to have a single output node for binary classification, suitable for
    BCEWithLogitsLoss.
    """

    def __init__(self, model_name: str, pretrained: bool = True):
        """
        Args:
            model_name (str): The name of the model architecture to load via timm.
            pretrained (bool): Whether to load pretrained weights (e.g., ImageNet).
        """
        super(PetModel, self).__init__()

        # Instantiate the model using timm.
        # We explicitly set num_classes=1 to create a binary classification head
        # (outputting a single logit), as specified in the task requirements.
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Height, Width).

        Returns:
            torch.Tensor: Logits of shape (Batch, 1).
        """
        return self.model(x)
