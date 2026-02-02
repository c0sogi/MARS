import torch
import torch.nn as nn
import timm
from library.config import Config


class AppleClassifier(nn.Module):
    """
    Apple Disease Detection Model.

    Architecture:
    - Backbone: EfficientNet-B5 (Noisy Student weights) via timm.
    - Pooling: Global Average Pooling (handled by timm).
    - Head: Dropout (p=0.4) -> Linear (4 classes).
    """

    def __init__(self, model_name=Config.MODEL_NAME, pretrained=True):
        """
        Args:
            model_name (str): Name of the timm model to load.
            pretrained (bool): Whether to load pretrained weights.
        """
        super(AppleClassifier, self).__init__()

        # Load the backbone
        # num_classes=0 removes the default FC layer.
        # global_pool='avg' ensures the output is a pooled feature vector (Batch, Num_Features).
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # Get the number of input features for the final layer
        in_features = self.backbone.num_features

        # Define the custom classification head
        self.dropout = nn.Dropout(p=Config.DROPOUT_RATE)
        self.fc = nn.Linear(in_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input images, shape (Batch, 3, H, W).

        Returns:
            torch.Tensor: Logits, shape (Batch, Num_Classes).
        """
        # Extract features using the backbone
        # Shape: (Batch, Num_Features) due to global_pool='avg'
        features = self.backbone(x)

        # Apply Dropout
        x = self.dropout(features)

        # Final Linear Layer
        logits = self.fc(x)

        return logits
