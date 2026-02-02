import torch
import torch.nn as nn
import timm
from library.config import Config


class DogClassifier(nn.Module):
    """
    Dog Breed Classifier using a ConvNeXt backbone.

    Attributes:
        backbone (nn.Module): The feature extractor (ConvNeXt).
        head (nn.Sequential): The classification head (Dropout + Linear).
    """

    def __init__(
        self,
        model_name=Config.model_name,
        num_classes=Config.num_classes,
        dropout_rate=Config.dropout_rate,
        pretrained=True,
    ):
        """
        Initializes the model.

        Args:
            model_name (str): Name of the timm model to load.
            num_classes (int): Number of target classes.
            dropout_rate (float): Dropout probability for the head.
            pretrained (bool): Whether to load ImageNet weights.
        """
        super(DogClassifier, self).__init__()

        # Load the backbone from timm.
        # num_classes=0 removes the default classifier and applies Global Average Pooling,
        # returning a feature vector (batch_size, num_features).
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )

        # Determine input features for the linear layer
        in_features = self.backbone.num_features

        # Define the custom classification head
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input batch of images.

        Returns:
            torch.Tensor: Logits for each class.
        """
        # Extract features using the backbone
        features = self.backbone(x)

        # Pass features through the classification head
        logits = self.head(features)

        return logits

    def set_backbone_trainable(self, trainable=True):
        """
        Freezes or unfreezes the backbone parameters for fine-tuning strategies.
        The classification head remains trainable regardless of this setting.

        Args:
            trainable (bool): If True, backbone parameters are updated during training.
                              If False, they are frozen.
        """
        for param in self.backbone.parameters():
            param.requires_grad = trainable

        # Ensure the head is always trainable
        for param in self.head.parameters():
            param.requires_grad = True
