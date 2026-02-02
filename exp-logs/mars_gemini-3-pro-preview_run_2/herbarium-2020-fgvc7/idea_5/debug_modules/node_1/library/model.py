import torch
import torch.nn as nn
import timm
from library.config import (
    MODEL_NAME,
    PRETRAINED,
    DROPOUT_RATE,
    EMBEDDING_SIZE,
    NUM_SPECIES_CLASSES,
)
from library.taxonomy import TaxonomyManager


class HierarchicalEfficientNet(nn.Module):
    """
    Hierarchical Multi-Task Learning Model based on EfficientNet-B3.

    This model uses a shared backbone to extract features and branches into three
    classification heads:
    1. Species Head (Fine-grained, Target)
    2. Genus Head (Medium-grained, Auxiliary)
    3. Family Head (Coarse-grained, Auxiliary)
    """

    def __init__(self, num_genus_classes=None, num_family_classes=None):
        """
        Args:
            num_genus_classes (int, optional): Number of genus classes. If None, fetched from TaxonomyManager.
            num_family_classes (int, optional): Number of family classes. If None, fetched from TaxonomyManager.
        """
        super(HierarchicalEfficientNet, self).__init__()

        # Dynamically determine class counts for auxiliary heads if not provided
        if num_genus_classes is None or num_family_classes is None:
            # We use load_cached_data=True assuming the training script has already prepared the cache
            taxonomy = TaxonomyManager(load_cached_data=True)
            if num_genus_classes is None:
                num_genus_classes = taxonomy.get_num_genus()
            if num_family_classes is None:
                num_family_classes = taxonomy.get_num_family()

        self.num_species = NUM_SPECIES_CLASSES
        self.num_genus = num_genus_classes
        self.num_family = num_family_classes

        # Load Pretrained Backbone
        # num_classes=0 removes the default classifier and returns the pooled feature vector
        self.backbone = timm.create_model(
            MODEL_NAME, pretrained=PRETRAINED, num_classes=0, global_pool="avg"
        )

        # Shared Dropout
        self.dropout = nn.Dropout(p=DROPOUT_RATE)

        # Hierarchical Heads
        # 1. Family Head (Coarse)
        self.head_family = nn.Linear(EMBEDDING_SIZE, self.num_family)

        # 2. Genus Head (Medium)
        self.head_genus = nn.Linear(EMBEDDING_SIZE, self.num_genus)

        # 3. Species Head (Fine - Target)
        self.head_species = nn.Linear(EMBEDDING_SIZE, self.num_species)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W)

        Returns:
            dict: Dictionary containing logits for 'species', 'genus', and 'family'.
        """
        # Extract features using the shared backbone
        # Shape: (Batch_Size, EMBEDDING_SIZE)
        features = self.backbone(x)

        # Apply dropout to the shared features
        features = self.dropout(features)

        # Parallel execution of hierarchical heads
        logits_family = self.head_family(features)
        logits_genus = self.head_genus(features)
        logits_species = self.head_species(features)

        return {
            "species": logits_species,
            "genus": logits_genus,
            "family": logits_family,
        }
