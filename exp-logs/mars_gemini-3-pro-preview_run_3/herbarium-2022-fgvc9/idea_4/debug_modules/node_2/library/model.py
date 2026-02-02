import torch
import torch.nn as nn
import timm
from library.config import Config
from library.utils import process_hierarchy_mappings


class HierarchicalEfficientNet(nn.Module):
    """
    Hierarchical Multi-Task Learning Model based on EfficientNetV2.

    This model uses a shared backbone to extract features and three parallel
    linear heads to predict taxonomic ranks:
    1. Species (Primary Task)
    2. Genus (Auxiliary Task)
    3. Family (Auxiliary Task)
    """

    def __init__(
        self, backbone_name, num_species, num_genera, num_families, pretrained=True
    ):
        """
        Args:
            backbone_name (str): Name of the timm backbone (e.g., 'tf_efficientnetv2_s').
            num_species (int): Number of species classes (primary target).
            num_genera (int): Number of genus classes.
            num_families (int): Number of family classes.
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super(HierarchicalEfficientNet, self).__init__()

        # Initialize backbone with global pooling (num_classes=0)
        # This returns the feature vector directly (Batch, Num_Features)
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )

        # Determine input features for the heads
        self.num_features = self.backbone.num_features

        # Define parallel heads for hierarchical classification
        self.head_species = nn.Linear(self.num_features, num_species)
        self.head_genus = nn.Linear(self.num_features, num_genera)
        self.head_family = nn.Linear(self.num_features, num_families)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images (Batch, Channels, Height, Width).

        Returns:
            dict: Dictionary containing logits for 'species', 'genus', and 'family'.
        """
        # Extract features
        features = self.backbone(x)

        # Compute logits for each task
        logits_species = self.head_species(features)
        logits_genus = self.head_genus(features)
        logits_family = self.head_family(features)

        return {
            "species": logits_species,
            "genus": logits_genus,
            "family": logits_family,
        }


def get_model(pretrained=True, load_cached_hierarchy=True):
    """
    Factory function to instantiate the HierarchicalEfficientNet.

    Automatically determines the number of classes for Genus and Family
    using the hierarchy mapping utility provided in library.utils.

    Args:
        pretrained (bool): Whether to load pretrained backbone weights.
        load_cached_hierarchy (bool): Whether to use cached hierarchy mappings.

    Returns:
        HierarchicalEfficientNet: The initialized model.
    """
    # Retrieve hierarchy mapping to determine auxiliary class counts
    # This uses the utility function from library.utils which handles caching
    hierarchy_df = process_hierarchy_mappings(
        Config.TRAIN_METADATA_JSON,
        Config.WORKING_DIR,
        load_cached_data=load_cached_hierarchy,
    )

    # Calculate number of unique classes for auxiliary tasks
    # IDs are 0-indexed integers, so the count is max_id + 1
    # We cast to int to ensure compatibility with nn.Linear
    num_genera = int(hierarchy_df["genus_id"].max()) + 1
    num_families = int(hierarchy_df["family_id"].max()) + 1
    num_species = Config.NUM_CLASSES

    # Instantiate model with the configuration parameters
    model = HierarchicalEfficientNet(
        backbone_name=Config.BACKBONE,
        num_species=num_species,
        num_genera=num_genera,
        num_families=num_families,
        pretrained=pretrained,
    )

    return model
