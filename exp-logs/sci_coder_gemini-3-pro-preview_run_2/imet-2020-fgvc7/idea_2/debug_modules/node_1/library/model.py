import torch
import torch.nn as nn
import timm
from library.config import Config


class ArtworkConvNeXt(nn.Module):
    """
    ArtworkConvNeXt model using a ConvNeXt-Tiny backbone.

    This class wraps a timm model, adapting it for the artwork attribute labeling task
    by configuring the output head to match the number of specific attributes (3474).
    """

    def __init__(self, model_name=None, pretrained=None, num_classes=None):
        """
        Args:
            model_name (str, optional): Name of the model architecture. Defaults to Config.MODEL_NAME.
            pretrained (bool, optional): Whether to load pretrained weights. Defaults to Config.PRETRAINED.
            num_classes (int, optional): Number of output classes. Defaults to Config.NUM_CLASSES.
        """
        super(ArtworkConvNeXt, self).__init__()

        # Resolve arguments with Config defaults if not provided
        self.model_name = model_name if model_name is not None else Config.MODEL_NAME
        self.pretrained = pretrained if pretrained is not None else Config.PRETRAINED
        self.num_classes = (
            num_classes if num_classes is not None else Config.NUM_CLASSES
        )

        # Create the model using timm
        # timm handles the backbone creation and head replacement automatically
        # when num_classes is specified.
        self.model = timm.create_model(
            self.model_name, pretrained=self.pretrained, num_classes=self.num_classes
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, Channels, Height, Width).

        Returns:
            torch.Tensor: Raw logits of shape (Batch_Size, Num_Classes).
        """
        return self.model(x)
