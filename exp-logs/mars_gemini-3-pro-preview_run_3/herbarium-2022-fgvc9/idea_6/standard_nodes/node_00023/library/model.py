import torch
import torch.nn as nn
import timm
from library.config import Config
from library.utils import get_hierarchy_mappings


class HierarchicalEfficientNet(nn.Module):
    """
    Hierarchical Multi-Task EfficientNetV2-Small model.

    This model uses a shared backbone to extract features and three parallel
    linear heads to predict Species (primary task), Genus (auxiliary task),
    and Family (auxiliary task).
    """

    def __init__(self, pretrained=True):
        """
        Initialize the model.

        Args:
            pretrained (bool): If True, loads ImageNet-1k pretrained weights for the backbone.
        """
        super(HierarchicalEfficientNet, self).__init__()

        # Retrieve the number of classes for auxiliary heads from the hierarchy mapping
        # We ignore the mapping dictionaries and keep the counts
        _, _, _, _, num_genera, num_families = get_hierarchy_mappings(
            load_cached_data=True
        )

        # Create the backbone using timm
        # num_classes=0 removes the default classifier and global pooling,
        # but for EfficientNet in timm, it typically returns the pooled features
        # if global_pool is not explicitly disabled. We want the pooled feature vector.
        self.backbone = timm.create_model(
            Config.MODEL_NAME, pretrained=pretrained, num_classes=0
        )

        # Get the input dimension for the linear heads
        in_features = self.backbone.num_features

        # Define the three hierarchical heads
        # Primary Head: Species
        self.head_species = nn.Linear(in_features, Config.NUM_CLASSES_SPECIES)

        # Auxiliary Head: Genus
        self.head_genus = nn.Linear(in_features, num_genera)

        # Auxiliary Head: Family
        self.head_family = nn.Linear(in_features, num_families)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input batch of images [Batch, Channels, Height, Width].

        Returns:
            dict: A dictionary containing logits for each task:
                - 'species': [Batch, NUM_CLASSES_SPECIES]
                - 'genus':   [Batch, num_genera]
                - 'family':  [Batch, num_families]
        """
        # Extract features from the backbone
        # Shape: [Batch, in_features]
        features = self.backbone(x)

        # Pass features through each head
        logits_species = self.head_species(features)
        logits_genus = self.head_genus(features)
        logits_family = self.head_family(features)

        # Return dictionary of logits
        return {
            "species": logits_species,
            "genus": logits_genus,
            "family": logits_family,
        }
