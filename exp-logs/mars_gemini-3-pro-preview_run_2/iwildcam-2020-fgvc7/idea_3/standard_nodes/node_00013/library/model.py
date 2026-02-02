import torch
import torch.nn as nn
import timm
from library.config import Config


class IWildCamModel(nn.Module):
    """
    IWildCamModel wraps a timm-based EfficientNetV2-M model for animal species classification.
    It handles the initialization of the backbone with ImageNet-21k weights,
    replaces the classification head, and moves the model to the configured device.
    """

    def __init__(self, model_name=None, num_classes=None, pretrained=True):
        """
        Args:
            model_name (str, optional): Name of the timm model to load. Defaults to Config.MODEL_NAME.
            num_classes (int, optional): Number of output classes. Defaults to Config.NUM_CLASSES.
            pretrained (bool, optional): Whether to load pretrained weights. Defaults to True.
        """
        super(IWildCamModel, self).__init__()

        # Use Config defaults if arguments are not provided
        self.model_name = model_name if model_name is not None else Config.MODEL_NAME
        self.num_classes = (
            num_classes if num_classes is not None else Config.NUM_CLASSES
        )

        # Create the model using timm
        # passing num_classes tells timm to replace the default head with a new Linear layer
        # initialized for the specific number of classes.
        self.backbone = timm.create_model(
            self.model_name, pretrained=pretrained, num_classes=self.num_classes
        )

        # Move the model to the computation device (GPU/CPU) defined in Config
        self.device = Config.DEVICE
        self.to(self.device)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, Channels, Height, Width).

        Returns:
            torch.Tensor: Logits of shape (Batch_Size, Num_Classes).
        """
        return self.backbone(x)
