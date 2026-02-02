import torch
import torch.nn as nn
import timm
from library.config import Config


class ArtworkModel(nn.Module):
    """
    ArtworkModel class based on ConvNeXt architecture.
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        num_classes=Config.NUM_CLASSES,
        pretrained=True,
    ):
        """
        Initializes the model.

        Args:
            model_name (str): Name of the model architecture in timm.
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to load pretrained weights.
        """
        super(ArtworkModel, self).__init__()

        # Load the backbone model
        # num_classes=0 removes the default classifier head and returns pooled features
        # global_pool='avg' ensures we get the Global Average Pooled features
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # Determine the input feature dimension for the head
        # For ConvNeXt-Base, this is typically 1024
        in_features = self.backbone.num_features

        # Define the new classification head
        # A simple Linear layer as specified in the strategy
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        # Extract features using the backbone
        # Shape: (B, in_features)
        features = self.backbone(x)

        # Project to class logits
        # Shape: (B, num_classes)
        logits = self.head(features)

        return logits
