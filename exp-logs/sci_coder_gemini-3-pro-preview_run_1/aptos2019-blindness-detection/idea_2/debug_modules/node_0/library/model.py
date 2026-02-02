import torch
import torch.nn as nn
import timm
from library.config import Config


class OrdinalEfficientNet(nn.Module):
    """
    EfficientNet-B4 with a Rank-Consistent Ordinal Regression Head.

    The model predicts K-1 binary probabilities for K classes.
    For Diabetic Retinopathy (5 classes: 0-4), it outputs 4 probabilities.
    Output unit k represents the probability P(y > k).
    """

    def __init__(self, backbone_name=Config.backbone, pretrained=Config.pretrained):
        """
        Args:
            backbone_name (str): Name of the timm model to load.
            pretrained (bool): Whether to load pretrained weights.
        """
        super(OrdinalEfficientNet, self).__init__()

        # Load the backbone model
        # num_classes=0 removes the top classification layer
        # global_pool='avg' ensures we get a flattened feature vector after pooling
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # Get the number of input features for the fully connected layer
        # timm models store this in num_features
        in_features = self.backbone.num_features

        # Ordinal Regression Head
        # We need num_ordinal_units (4) outputs for 5 classes
        self.fc = nn.Linear(in_features, Config.num_ordinal_units)

        # Sigmoid activation to squash outputs to probabilities [0, 1]
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W)

        Returns:
            torch.Tensor: Ordinal probabilities of shape (B, 4)
        """
        # Extract features using the backbone
        features = self.backbone(x)

        # Pass through the linear layer
        logits = self.fc(features)

        # Apply Sigmoid activation
        probs = self.activation(logits)

        return probs
