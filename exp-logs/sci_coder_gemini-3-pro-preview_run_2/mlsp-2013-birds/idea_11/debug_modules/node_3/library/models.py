import torch
import torch.nn as nn
import timm
from library.config import Config


class BirdClassifier(nn.Module):
    """
    A unified model class for the Heterogeneous Ensemble.
    Supports ResNet18, DenseNet121, and EfficientNet-B0 backbones via timm.

    Architecture:
    Input (3-channel) -> Backbone -> Global Average Pooling -> Linear(19) -> Logits
    """

    def __init__(self, backbone_name, num_classes=Config.NUM_CLASSES, pretrained=True):
        """
        Args:
            backbone_name (str): Name of the timm model (e.g., 'resnet18', 'densenet121').
            num_classes (int): Number of output classes (19 for this dataset).
            pretrained (bool): Whether to load ImageNet pretrained weights.
        """
        super(BirdClassifier, self).__init__()

        # Create the backbone using timm
        # num_classes=0 removes the top fully connected layer
        # global_pool='avg' enforces Global Average Pooling, strictly avoiding Max/GeM pooling
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            in_chans=Config.CHANNELS,
        )

        # Retrieve the number of features output by the backbone after pooling
        in_features = self.backbone.num_features

        # Define the single fully connected layer as the classification head
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 3, Height, Width)

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes)
        """
        # Extract features (includes GAP due to global_pool='avg' in create_model)
        features = self.backbone(x)

        # Flatten is usually handled by timm's pooling, but we ensure shape consistency
        # features shape: [Batch, num_features]

        # Pass through linear head
        logits = self.fc(features)

        return logits


def get_model(
    backbone_name, num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED
):
    """
    Factory function to instantiate a BirdClassifier.

    Args:
        backbone_name (str): Name of the backbone architecture.
        num_classes (int): Number of target classes.
        pretrained (bool): Whether to use pretrained weights.

    Returns:
        BirdClassifier: The initialized PyTorch model.
    """
    return BirdClassifier(backbone_name, num_classes=num_classes, pretrained=pretrained)
