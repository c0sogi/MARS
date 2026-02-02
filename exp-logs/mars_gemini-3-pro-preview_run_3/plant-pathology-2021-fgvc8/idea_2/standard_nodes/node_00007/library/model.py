import torch
import torch.nn as nn
import timm
from library.config import Config


class AppleClassifier(nn.Module):
    """
    Apple Disease Detection Model based on EfficientNet-B0.

    Uses timm to load the backbone and configure the classification head.
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        num_classes=Config.NUM_CLASSES,
        pretrained=Config.PRETRAINED,
        dropout_rate=Config.DROPOUT_RATE,
    ):
        """
        Args:
            model_name (str): Name of the model architecture (e.g., 'efficientnet_b0').
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to load pretrained ImageNet weights.
            dropout_rate (float): Dropout rate for the classification head.
        """
        super(AppleClassifier, self).__init__()

        # Create the model using timm
        # timm automatically handles the global average pooling and linear classification head
        # when num_classes is specified.
        # drop_rate controls the dropout before the final classifier.
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout_rate,
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        return self.model(x)
