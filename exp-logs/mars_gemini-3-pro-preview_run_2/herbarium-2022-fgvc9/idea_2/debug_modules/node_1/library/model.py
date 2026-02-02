import torch
import torch.nn as nn
import timm
from library.config import Config


class HierarchicalEfficientNet(nn.Module):
    """
    Hierarchical Multi-Task Network architecture for Plant Classification.
    Uses an EfficientNet-B3 backbone to extract features, which are then fed
    into three parallel classification heads for Family, Genus, and Species.
    """

    def __init__(
        self, num_families, num_genera, num_species=Config.NUM_CLASSES, pretrained=True
    ):
        """
        Args:
            num_families (int): Number of unique family classes.
            num_genera (int): Number of unique genus classes.
            num_species (int): Number of unique species classes. Defaults to Config.NUM_CLASSES.
            pretrained (bool): Whether to load ImageNet pretrained weights. Defaults to True.
        """
        super(HierarchicalEfficientNet, self).__init__()

        # Load EfficientNet-B3 backbone
        # num_classes=0 removes the default classifier and returns the pooled feature vector
        # global_pool='avg' ensures Global Average Pooling is applied
        self.backbone = timm.create_model(
            Config.MODEL_NAME, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # Retrieve the number of output features from the backbone
        in_features = self.backbone.num_features

        # Dropout for regularization before the classification heads
        self.dropout = nn.Dropout(p=Config.DROPOUT_RATE)

        # Define parallel classification heads
        self.family_head = nn.Linear(in_features, num_families)
        self.genus_head = nn.Linear(in_features, num_genera)
        self.species_head = nn.Linear(in_features, num_species)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input batch of images [Batch, Channels, Height, Width].

        Returns:
            dict: A dictionary containing logits for each taxonomic level:
                  - 'species': Logits for species classification
                  - 'genus': Logits for genus classification
                  - 'family': Logits for family classification
        """
        # Extract features from the backbone
        # Shape: [Batch, in_features]
        features = self.backbone(x)

        # Apply dropout
        features = self.dropout(features)

        # Compute logits for each task
        family_logits = self.family_head(features)
        genus_logits = self.genus_head(features)
        species_logits = self.species_head(features)

        return {
            "species": species_logits,
            "genus": genus_logits,
            "family": family_logits,
        }
