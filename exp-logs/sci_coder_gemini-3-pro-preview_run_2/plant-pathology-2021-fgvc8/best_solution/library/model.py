import torch
import torch.nn as nn
import timm
from library.config import Config


class AppleDiseaseModel(nn.Module):
    """
    Neural Network model for Apple Disease Detection.
    Uses a ConvNeXt backbone with a custom linear head for multi-label classification.
    """

    def __init__(
        self,
        model_name: str = Config.MODEL_NAME,
        pretrained: bool = Config.PRETRAINED,
        num_classes: int = Config.NUM_CLASSES,
        drop_rate: float = Config.DROP_RATE,
        drop_path_rate: float = Config.DROP_PATH_RATE,
    ):
        """
        Initializes the model.

        Args:
            model_name (str): The name of the timm model to use as backbone.
            pretrained (bool): Whether to use pretrained weights.
            num_classes (int): Number of output classes.
            drop_rate (float): Dropout rate for the model.
            drop_path_rate (float): Drop path rate (stochastic depth) for the model.
        """
        super(AppleDiseaseModel, self).__init__()

        # Initialize the backbone using timm
        # num_classes=0 removes the default classifier head
        # global_pool='avg' ensures the output is a pooled feature vector
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

        # Get the number of input features for the final layer
        in_features = self.backbone.num_features

        # Define the custom classification head
        # Simple linear projection as per the strategy
        self.fc = nn.Linear(in_features, num_classes)

        # Define the loss function
        # BCEWithLogitsLoss is suitable for multi-label classification
        # It combines a Sigmoid layer and the BCELoss in one single class
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input batch of images (B, C, H, W).

        Returns:
            torch.Tensor: Raw logits (B, num_classes).
        """
        # Extract features from the backbone
        features = self.backbone(x)

        # Pass through the classification head
        logits = self.fc(features)

        return logits

    def get_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss between predictions and targets.

        Args:
            logits (torch.Tensor): Predicted logits from the forward pass.
            targets (torch.Tensor): Ground truth labels (multi-hot encoded).

        Returns:
            torch.Tensor: The computed loss value.
        """
        return self.loss_fn(logits, targets)
