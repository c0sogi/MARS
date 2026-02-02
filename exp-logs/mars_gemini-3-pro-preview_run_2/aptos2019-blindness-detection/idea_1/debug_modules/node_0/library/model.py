import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from library.config import Config


class RetinopathyRegressor(nn.Module):
    """
    A regression model based on MobileNetV3-Small for predicting Diabetic Retinopathy severity.
    The model outputs a continuous scalar score representing the severity level.
    """

    def __init__(self, pretrained: bool = True):
        """
        Initialize the model architecture.

        Args:
            pretrained (bool): If True, initializes the backbone with ImageNet pre-trained weights.
        """
        super(RetinopathyRegressor, self).__init__()

        # Load the MobileNetV3-Small backbone
        # using the modern torchvision weights API
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.model = mobilenet_v3_small(weights=weights)

        # The classifier in MobileNetV3-Small is a Sequential block.
        # Standard structure:
        # (0): Linear(in_features=576, out_features=1024)
        # (1): Hardswish()
        # (2): Dropout(p=0.2)
        # (3): Linear(in_features=1024, out_features=1000)

        # We target the final linear layer (index 3) to change the output dimension.
        classifier = self.model.classifier
        last_layer_index = 3

        # Get the input features of the last layer
        in_features = classifier[last_layer_index].in_features

        # Replace the classification head with a regression head
        # Config.NUM_CLASSES is 1 for regression
        classifier[last_layer_index] = nn.Linear(in_features, Config.NUM_CLASSES)

        # Initialize the new layer's weights randomly
        # Xavier uniform is a good default for linear layers
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
