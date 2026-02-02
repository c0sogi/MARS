import torch
import torch.nn as nn
import timm
from library.config import Config
from library.layers import GeM


class BirdClassifier(nn.Module):
    """
    Bird Species Classifier using a heterogeneous backbone with GeM Pooling.

    Architecture:
    1. Pretrained Backbone (ResNet, DenseNet, or EfficientNet) -> Returns spatial features (B, C, H, W)
    2. Generalized Mean Pooling (GeM) -> Adaptive pooling (B, C, 1, 1)
    3. Flatten -> (B, C)
    4. Linear Head -> (B, num_classes)
    """

    def __init__(self, backbone_name, num_classes=Config.NUM_CLASSES, pretrained=True):
        """
        Args:
            backbone_name (str): Name of the timm model (e.g., 'resnet18', 'densenet121').
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to load ImageNet pretrained weights.
        """
        super(BirdClassifier, self).__init__()

        # Create the backbone model
        # num_classes=0 and global_pool='' ensures we get the last convolutional feature map
        # without the default pooling or classification head.
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Determine the number of input features for the linear head
        # Most timm models have a num_features attribute
        if hasattr(self.backbone, "num_features"):
            in_features = self.backbone.num_features
        else:
            # Fallback: perform a dummy forward pass to infer shape
            with torch.no_grad():
                # Create a dummy input with the configured image size (Channels=3)
                dummy_input = torch.randn(1, 3, *Config.IMAGE_SIZE)
                features = self.backbone(dummy_input)
                in_features = features.shape[1]

        # Initialize the custom GeM pooling layer
        self.global_pool = GeM()

        # Classification Head
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
        # Shape: (Batch, Channels, Height_feat, Width_feat)
        x = self.backbone(x)

        # Apply Generalized Mean Pooling
        # Shape: (Batch, Channels, 1, 1)
        x = self.global_pool(x)

        # Flatten for the linear layer
        # Shape: (Batch, Channels)
        x = x.flatten(1)

        # Compute logits
        logits = self.fc(x)

        return logits


def get_model(backbone_name, num_classes=Config.NUM_CLASSES, pretrained=True):
    """
    Factory function to instantiate the BirdClassifier.

    Args:
        backbone_name (str): Name of the backbone architecture.
        num_classes (int): Number of target classes.
        pretrained (bool): Use pretrained weights.

    Returns:
        BirdClassifier: The initialized model.
    """
    return BirdClassifier(backbone_name, num_classes=num_classes, pretrained=pretrained)
