import torch
import torch.nn as nn
import timm
from library.config import Config


class AppleDiseaseModel(nn.Module):
    """
    Neural Network model for Apple Disease Detection.
    Uses an EfficientNet-B0 backbone with a custom classification head.
    """

    def __init__(
        self,
        model_name: str = Config.MODEL_NAME,
        pretrained: bool = True,
        num_classes: int = Config.NUM_CLASSES,
        dropout_rate: float = Config.DROPOUT_RATE,
    ):
        """
        Initializes the AppleDiseaseModel.

        Args:
            model_name (str): The name of the backbone model architecture (default: efficientnet_b0).
            pretrained (bool): Whether to use pretrained weights (default: True).
            num_classes (int): The number of target classes (default: 4).
            dropout_rate (float): The dropout rate for the classification head (default: 0.2).
        """
        super(AppleDiseaseModel, self).__init__()

        # Create the backbone model using timm
        # num_classes=0 removes the default classifier
        # global_pool='avg' ensures we get a feature vector (Global Average Pooling)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # Retrieve the number of input features for the classifier
        # EfficientNet-B0 typically has 1280 features after GAP
        self.in_features = self.backbone.num_features

        # Define the custom classification head
        # Consists of a Dropout layer for regularization and a Linear layer for prediction
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(self.in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the network.

        Args:
            x (torch.Tensor): Input batch of images. Shape: (B, C, H, W).

        Returns:
            torch.Tensor: Raw logits for each class. Shape: (B, Num_Classes).
        """
        # Extract features from the backbone
        features = self.backbone(x)

        # Pass features through the classifier head
        logits = self.classifier(features)

        return logits
