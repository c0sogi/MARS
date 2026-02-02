import torch
import torch.nn as nn
import timm
from library.config import Config


class EfficientNetRegressor(nn.Module):
    """
    EfficientNet-based regression model for Diabetic Retinopathy severity prediction.

    This architecture uses a pretrained EfficientNet-B3 backbone to extract features,
    followed by a regression head to predict a continuous severity score.
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        pretrained=Config.PRETRAINED,
        dropout_rate=Config.DROPOUT_RATE,
    ):
        """
        Initialize the EfficientNetRegressor.

        Args:
            model_name (str): Name of the backbone model to load from timm.
            pretrained (bool): Whether to use pretrained ImageNet weights.
            dropout_rate (float): Probability for the Dropout layer.
        """
        super(EfficientNetRegressor, self).__init__()

        # Load the backbone using timm
        # num_classes=0 removes the default classifier and returns the pooled features
        # global_pool='avg' ensures we get the Global Average Pooled output
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # Retrieve the number of input features for the final linear layer
        # timm models store this in num_features
        in_features = self.backbone.num_features

        # Define the regression head
        # 1. Dropout for regularization to prevent overfitting
        self.dropout = nn.Dropout(p=dropout_rate)

        # 2. Single Linear layer to output a scalar regression score
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input batch of images, shape (B, C, H, W).

        Returns:
            torch.Tensor: Predicted scalar severity scores, shape (B, 1).
        """
        # Extract features from the backbone (includes Global Average Pooling)
        # Shape: (B, num_features)
        features = self.backbone(x)

        # Apply Dropout
        features = self.dropout(features)

        # Pass through the linear regression layer
        # Shape: (B, 1)
        out = self.fc(features)

        return out
