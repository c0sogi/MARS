import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class TumorClassifier(nn.Module):
    """
    TumorClassifier based on ResNet18 architecture.

    This model uses a pre-trained ResNet18 backbone and replaces the
    final classification layer to output a single logit for binary classification
    (Tumor vs No Tumor).
    """

    def __init__(
        self,
        pretrained: bool = Config.PRETRAINED,
        dropout_rate: float = Config.DROPOUT_RATE,
    ):
        """
        Initializes the TumorClassifier.

        Args:
            pretrained (bool): If True, loads ImageNet pre-trained weights.
            dropout_rate (float): Dropout probability for the classification head.
        """
        super(TumorClassifier, self).__init__()

        # Load the model
        weights = "DEFAULT" if pretrained else None
        if Config.MODEL_NAME == "resnet34":
            self.model = models.resnet34(weights=weights)
        else:
            self.model = models.resnet18(weights=weights)

        # ResNet structure ends with a Linear layer named 'fc'.
        # We replace it with a Sequential block to include Dropout and the final Linear layer.
        in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(in_features, Config.NUM_CLASSES)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output logits of shape (B, 1).
        """
        return self.model(x)
