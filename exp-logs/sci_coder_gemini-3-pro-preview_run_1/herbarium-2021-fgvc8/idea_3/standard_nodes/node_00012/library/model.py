import torch
import torch.nn as nn
import timm
from library.config import Config


class HierarchicalEfficientNet(nn.Module):
    """
    Hierarchical Multi-Task Learning Model based on EfficientNet-B0.

    This architecture leverages a shared backbone to extract visual features,
    which are then fed into two parallel heads:
    1. Species Head: Predicts the fine-grained plant species (Primary Task).
    2. Family Head: Predicts the coarse-grained plant family (Auxiliary Task).

    The auxiliary family task acts as a regularizer, helping the model learn
    robust features for rare species by sharing information across the taxonomic hierarchy.
    """

    def __init__(
        self, num_families, num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED
    ):
        """
        Args:
            num_families (int): Number of unique plant families (size of auxiliary output).
            num_classes (int): Number of unique plant species (size of primary output).
            pretrained (bool): Whether to initialize the backbone with ImageNet weights.
        """
        super(HierarchicalEfficientNet, self).__init__()

        # Initialize the EfficientNet-B0 backbone.
        # Setting num_classes=0 removes the default classification layer and
        # returns the global average pooled features (flattened).
        self.backbone = timm.create_model(
            Config.BACKBONE, pretrained=pretrained, num_classes=0
        )

        # Get the dimension of the feature vector output by the backbone.
        # For EfficientNet-B0, this is typically 1280.
        self.in_features = self.backbone.num_features

        # Primary Head: Projects features to species logits
        self.species_head = nn.Linear(self.in_features, num_classes)

        # Auxiliary Head: Projects features to family logits
        self.family_head = nn.Linear(self.in_features, num_families)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input batch of images with shape (Batch, Channels, Height, Width).

        Returns:
            tuple: A tuple containing:
                - species_logits (torch.Tensor): Output for species classification.
                - family_logits (torch.Tensor): Output for family classification.
        """
        # Extract features using the shared backbone
        # Shape: (Batch_Size, In_Features)
        features = self.backbone(x)

        # Compute outputs for both tasks
        species_logits = self.species_head(features)
        family_logits = self.family_head(features)

        return species_logits, family_logits
