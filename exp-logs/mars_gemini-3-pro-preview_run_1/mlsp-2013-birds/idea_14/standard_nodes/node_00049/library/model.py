import torch
import torch.nn as nn
import timm
from library.config import Config


class SEResNet34(nn.Module):
    """
    SE-ResNet-34 model for Bird Species Classification.

    This model uses a Squeeze-and-Excitation ResNet-34 backbone initialized with
    ImageNet weights. The default classification head is replaced with a simple
    Linear layer to map pooled features to the species logits.
    """

    def __init__(self, pretrained=Config.PRETRAINED):
        """
        Initialize the model.

        Args:
            pretrained (bool): Whether to load ImageNet pretrained weights.
                               Defaults to Config.PRETRAINED.
        """
        super(SEResNet34, self).__init__()

        # Load the SE-ResNet-34 backbone from timm
        # num_classes=0: Removes the default classification head and returns the
        #                output of the global pooling layer.
        # global_pool='avg': Ensures Global Average Pooling is applied.
        self.backbone = timm.create_model(
            Config.MODEL_NAME, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # Retrieve the number of input features for the classification head
        in_features = self.backbone.num_features

        # Define the classification head
        # A simple Linear layer projecting features to the number of classes (19)
        self.fc = nn.Linear(in_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images of shape (Batch, 3, Height, Width).

        Returns:
            torch.Tensor: Raw logits of shape (Batch, Num_Classes).
        """
        # Pass input through the backbone to get pooled features
        # Shape: (Batch, in_features)
        features = self.backbone(x)

        # Pass features through the linear classification head
        # Shape: (Batch, Num_Classes)
        logits = self.fc(features)

        return logits
