import torch
import torch.nn as nn
import timm
from library.config import SEED, DROPOUT
from library.utils import seed_everything

# Ensure reproducibility
seed_everything(SEED)


class EfficientNetExpert(nn.Module):
    """
    EfficientNet-B0 based Expert Model for Spatially-Decomposed Consensus.
    Wraps timm's implementation to allow easy configuration of dropout and output classes.
    """

    def __init__(
        self,
        model_name="efficientnet_b0",
        pretrained=True,
        num_classes=1,
        dropout_rate=DROPOUT,
    ):
        """
        Args:
            model_name (str): Name of the model architecture in timm.
            pretrained (bool): Whether to load ImageNet pretrained weights.
            num_classes (int): Number of output classes (1 for binary classification logits).
            dropout_rate (float): Dropout rate for the classifier head.
        """
        super(EfficientNetExpert, self).__init__()

        # Create the model using timm
        # drop_rate controls the dropout in the classifier head
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout_rate,
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        return self.model(x)
