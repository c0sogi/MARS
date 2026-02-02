import torch
import torch.nn as nn
import timm
from library import config


class MGMTNet(nn.Module):
    """
    High-Resolution ROI-Cropped Content-Anchored (HRC-CA) Network Model.
    Wraps an EfficientNet-B0 backbone for binary classification of MGMT promoter methylation.
    """

    def __init__(
        self,
        model_name=config.MODEL_NAME,
        pretrained=config.PRETRAINED,
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE,
        in_channels=config.NUM_CHANNELS,
    ):
        """
        Args:
            model_name (str): Name of the backbone model (default: 'efficientnet_b0').
            pretrained (bool): Whether to load ImageNet weights.
            num_classes (int): Number of output classes (1 for binary classification).
            dropout_rate (float): Dropout probability for the classifier head.
            in_channels (int): Number of input channels (default: 3).
        """
        super(MGMTNet, self).__init__()

        # Create the backbone model
        # num_classes=0 removes the default classifier head and pooling,
        # but we want the pooling to handle feature extraction cleanly.
        # global_pool='avg' ensures we get a flat vector (Batch, Num_Features)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove default head
            in_chans=in_channels,
            global_pool="avg",
        )

        # Get the number of input features for the classifier
        # For EfficientNet-B0, this is typically 1280
        in_features = self.backbone.num_features

        # Define the custom classifier head
        # Structure: Dropout -> Linear
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Height, Width).

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes).
        """
        # Extract features using the backbone
        # Output shape: (Batch, Num_Features)
        features = self.backbone(x)

        # Pass features through the classifier head
        # Output shape: (Batch, Num_Classes)
        logits = self.classifier(features)

        return logits
