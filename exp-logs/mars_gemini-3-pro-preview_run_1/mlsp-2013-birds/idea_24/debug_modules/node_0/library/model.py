import torch
import torch.nn as nn
import timm
from library.config import Config


class BirdClassifier(nn.Module):
    """
    Bird Species Classifier architecture based on ResNet34-d.

    This class implements the structural innovation of using a deep-stem ResNet
    to preserve spectral fidelity. It uses a standard linear classification head
    to map extracted features to species probabilities.
    """

    def __init__(self, pretrained=True):
        """
        Args:
            pretrained (bool): If True, initializes the backbone with ImageNet weights.
        """
        super(BirdClassifier, self).__init__()

        # Initialize the backbone using timm
        # Config.BACKBONE is set to 'resnet34d'
        # num_classes=0 removes the default head
        # global_pool='avg' ensures the output is a pooled feature vector
        self.backbone = timm.create_model(
            Config.BACKBONE, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # Retrieve the number of input features from the backbone
        # For ResNet34, this is typically 512
        in_features = self.backbone.num_features

        # Define the Classification Head
        # A simple Linear Layer projecting to the 19 species classes
        # This avoids complex aggregation heads as per the design requirements
        self.fc = nn.Linear(in_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, Channels, Height, Width).
                              Expected resolution is 256x640.

        Returns:
            torch.Tensor: Logits of shape (Batch_Size, Num_Classes).
        """
        # Feature Extraction
        features = self.backbone(x)

        # Ensure features are flattened (Batch_Size, Num_Features)
        # timm with global_pool='avg' usually returns 2D tensor, but explicit flattening is safe
        features = features.view(features.size(0), -1)

        # Classification
        logits = self.fc(features)

        return logits


def create_model(pretrained=True):
    """
    Factory function to create an instance of the BirdClassifier.

    Args:
        pretrained (bool): Whether to use pretrained weights for the backbone.

    Returns:
        BirdClassifier: The initialized model.
    """
    model = BirdClassifier(pretrained=pretrained)
    return model
