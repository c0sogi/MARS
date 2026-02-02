import torch
import torch.nn as nn
import timm
from library.config import MODEL_NAME, PRETRAINED, DROPOUT_RATE, NUM_CLASSES


class VAANet(nn.Module):
    """
    Verified Anatomically-Anchored (VAA) Network.

    Architecture:
    - Backbone: EfficientNet-B0 (initialized with ImageNet weights).
    - Input: 3-Channel 2D Images (FLAIR, T1wCE, T2w).
    - Head: Dropout (0.3) -> Linear Output (1 class).
    """

    def __init__(self):
        super(VAANet, self).__init__()

        # Initialize the backbone using timm
        # num_classes=0 removes the default fully connected layer
        # global_pool='avg' ensures we get a flattened feature vector
        self.backbone = timm.create_model(
            MODEL_NAME, pretrained=PRETRAINED, num_classes=0, global_pool="avg"
        )

        # Retrieve the number of features output by the backbone
        # For EfficientNet-B0, this is typically 1280
        in_features = self.backbone.num_features

        # Define the custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT_RATE), nn.Linear(in_features, NUM_CLASSES)
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 3, Height, Width)

        Returns:
            torch.Tensor: Logits of shape (Batch, 1)
        """
        # Extract features from the backbone
        # Shape: (Batch, in_features)
        features = self.backbone(x)

        # Pass features through the custom classifier
        # Shape: (Batch, 1)
        logits = self.classifier(features)

        return logits


def build_model():
    """
    Factory function to instantiate the VAANet model.

    Returns:
        VAANet: An instance of the model.
    """
    model = VAANet()
    return model
