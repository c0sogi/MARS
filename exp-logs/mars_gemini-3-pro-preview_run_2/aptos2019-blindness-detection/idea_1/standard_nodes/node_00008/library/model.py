import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from library.config import Config


class RetinopathyRegressor(nn.Module):
    """
    A regression model based on EfficientNet-B3 for predicting Diabetic Retinopathy severity.
    The model outputs a continuous scalar score representing the severity level.
    """

    def __init__(self, pretrained: bool = True):
        """
        Initialize the model architecture.

        Args:
            pretrained (bool): If True, initializes the backbone with ImageNet pre-trained weights.
        """
        super(RetinopathyRegressor, self).__init__()

        # Load the EfficientNet-B3 backbone
        weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
        self.model = efficientnet_b3(weights=weights)

        # The classifier in EfficientNet-B3 is a Sequential block:
        # (0): Dropout(p=0.2, inplace=True)
        # (1): Linear(in_features=1280, out_features=1000, bias=True)

        classifier = self.model.classifier
        last_layer_index = 1

        # Get the input features of the last layer
        in_features = classifier[last_layer_index].in_features

        # Replace the classification head with a regression head
        classifier[last_layer_index] = nn.Linear(in_features, Config.NUM_CLASSES)

        # Initialize the new layer's weights randomly
        nn.init.xavier_uniform_(classifier[last_layer_index].weight)
        if classifier[last_layer_index].bias is not None:
            nn.init.zeros_(classifier[last_layer_index].bias)

        # Reassign the modified classifier back to the model
        self.model.classifier = classifier

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, Channels, Height, Width).

        Returns:
            torch.Tensor: Regression scores of shape (Batch_Size,).
        """
        # Pass through the MobileNetV3 architecture
        out = self.model(x)

        # The output shape is (Batch_Size, 1).
        # Flatten to (Batch_Size,) to align with standard regression target shapes.
        return out.view(-1)
