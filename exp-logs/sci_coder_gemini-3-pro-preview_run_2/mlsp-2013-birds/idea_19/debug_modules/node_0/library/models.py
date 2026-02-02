import torch
import torch.nn as nn
import timm


class BirdModel(nn.Module):
    """
    A unified model class for the Tri-Architecture Heterogeneous Ensemble.
    Supports ResNet18, EfficientNet-B0, and DenseNet121 backbones.

    Implements a specific head design: Global Average Pooling (GAP) -> Linear.
    """

    def __init__(self, backbone_name, num_classes=19, pretrained=True):
        """
        Args:
            backbone_name (str): Name of the backbone ('resnet18', 'efficientnet_b0', 'densenet121').
            num_classes (int): Number of output classes (19 for bird species).
            pretrained (bool): Whether to load ImageNet pretrained weights.
        """
        super(BirdModel, self).__init__()
        self.backbone_name = backbone_name

        # Create the backbone using timm
        # num_classes=0 removes the fully connected layer
        # global_pool='' removes the default pooling layer, returning feature maps (B, C, H, W)
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Determine the number of input features for the linear layer
        # We do this by passing a dummy input through the backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            in_features = features.shape[1]

        # Define the custom head
        # 1. Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 2. Single Fully Connected Layer
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        # Extract features from backbone
        # Shape: (B, C, H_feat, W_feat)
        x = self.backbone(x)

        # Apply Global Average Pooling
        # Shape: (B, C, 1, 1)
        x = self.global_pool(x)

        # Flatten
        # Shape: (B, C)
        x = torch.flatten(x, 1)

        # Classification layer
        # Shape: (B, num_classes)
        logits = self.fc(x)

        return logits


def get_model(backbone_name, num_classes=19, pretrained=True):
    """
    Factory function to create a BirdModel instance.

    Args:
        backbone_name (str): 'resnet18', 'efficientnet_b0', or 'densenet121'.
        num_classes (int): Number of target classes.
        pretrained (bool): Use pretrained weights.

    Returns:
        BirdModel: The initialized model.
    """
    # Validate backbone name to ensure it matches the ensemble strategy
    valid_backbones = ["resnet18", "efficientnet_b0", "densenet121"]
    if backbone_name not in valid_backbones:
        raise ValueError(
            f"Backbone '{backbone_name}' is not supported. Choose from {valid_backbones}."
        )

    return BirdModel(backbone_name, num_classes=num_classes, pretrained=pretrained)
