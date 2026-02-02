import torch
import torch.nn as nn
import timm
from library.config import Config


class BirdClassifier(nn.Module):
    """
    Bird Species Classifier using a heterogeneous backbone with Global Average Pooling (GAP).
    Cite solution_lesson_node_00039: GAP acts as a powerful regularizer in small-data regimes.

    Architecture:
    1. Pretrained Backbone (ResNet, DenseNet) -> Returns pooled features (B, C)
    2. Linear Head -> (B, num_classes)
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
        # global_pool='avg' ensures we get the Global Average Pooled features directly
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # Determine the number of input features for the linear head
        if hasattr(self.backbone, "num_features"):
            in_features = self.backbone.num_features
        else:
            # Fallback: perform a dummy forward pass to infer shape
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, *Config.IMAGE_SIZE)
                features = self.backbone(dummy_input)
                in_features = features.shape[1]

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
        # Shape: (Batch, Channels) due to global_pool="avg"
        x = self.backbone(x)

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
