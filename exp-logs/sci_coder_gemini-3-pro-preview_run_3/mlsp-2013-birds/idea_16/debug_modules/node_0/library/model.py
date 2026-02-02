import torch
import torch.nn as nn
import timm
from library.config import Config


class BirdClassifier(nn.Module):
    """
    Bird Species Classifier architecture.

    Implements a heterogeneous backbone factory using `timm` and a
    Multi-Sample Dropout (MSD) classification head for robust feature learning.
    """

    def __init__(self, backbone_name: str, num_classes: int, pretrained: bool = True):
        """
        Args:
            backbone_name (str): Name of the backbone architecture (e.g., 'resnet18').
            num_classes (int): Number of target classes.
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super(BirdClassifier, self).__init__()

        # Initialize backbone using timm
        # num_classes=0 removes the default head
        # global_pool='avg' ensures we get a flattened feature vector
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            in_chans=Config.CHANNELS,
        )

        # Retrieve the number of output features from the backbone
        self.in_features = self.backbone.num_features

        # Multi-Sample Dropout (MSD) Module
        # Initialize K parallel dropout layers with different masks
        self.dropouts = nn.ModuleList(
            [
                nn.Dropout(p=Config.MSD_DROPOUT_RATE)
                for _ in range(Config.MSD_NUM_SAMPLES)
            ]
        )

        # Single shared linear classification layer
        self.fc = nn.Linear(self.in_features, num_classes)

    def forward(self, x):
        """
        Forward pass with Multi-Sample Dropout aggregation.

        Args:
            x (torch.Tensor): Input images, shape (Batch, Channels, Height, Width).

        Returns:
            torch.Tensor: Averaged logits, shape (Batch, Num_Classes).
        """
        # Extract features from the backbone
        # Shape: (Batch, In_Features)
        features = self.backbone(x)

        # Pass features through each dropout layer and the shared linear layer
        logits_list = []
        for dropout_layer in self.dropouts:
            # Apply dropout
            dropped_features = dropout_layer(features)
            # Apply linear layer
            logits = self.fc(dropped_features)
            logits_list.append(logits)

        # Stack the logits from all dropout samples
        # Shape: (Num_Samples, Batch, Num_Classes)
        stacked_logits = torch.stack(logits_list, dim=0)

        # Average the logits to produce the final prediction
        # Shape: (Batch, Num_Classes)
        mean_logits = torch.mean(stacked_logits, dim=0)

        return mean_logits
