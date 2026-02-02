import torch
import torch.nn as nn
import timm


class HierarchicalConvNeXt(nn.Module):
    """
    Hierarchical ConvNeXt-Tiny model for plant classification.

    This architecture uses a ConvNeXt-Tiny backbone to extract features, which are then
    fed into three parallel linear heads to predict Family, Genus, and Species simultaneously.
    This supports a hierarchical multi-task learning strategy.
    """

    def __init__(
        self, num_families=272, num_genera=2564, num_species=15501, pretrained=True
    ):
        """
        Args:
            num_families (int): Number of unique plant families (default: 272).
            num_genera (int): Number of unique plant genera (default: 2564).
            num_species (int): Number of unique plant species (default: 15501).
            pretrained (bool): If True, initializes the backbone with ImageNet-1k weights.
        """
        super(HierarchicalConvNeXt, self).__init__()

        # Initialize the ConvNeXt-Tiny backbone.
        # num_classes=0 ensures the model returns the global pooled feature vector
        # instead of a classification logit vector.
        self.backbone = timm.create_model(
            "convnext_tiny", pretrained=pretrained, num_classes=0
        )

        # Retrieve the embedding dimension (e.g., 768 for ConvNeXt-Tiny)
        embedding_dim = self.backbone.num_features

        # Define the three parallel classification heads
        self.family_head = nn.Linear(embedding_dim, num_families)
        self.genus_head = nn.Linear(embedding_dim, num_genera)
        self.species_head = nn.Linear(embedding_dim, num_species)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input batch of images with shape (Batch, Channels, Height, Width).

        Returns:
            dict: A dictionary containing the logits for each task:
                - 'family': Tensor of shape (Batch, num_families)
                - 'genus': Tensor of shape (Batch, num_genera)
                - 'species': Tensor of shape (Batch, num_species)
        """
        # Extract features from the backbone
        # Output shape: (Batch, embedding_dim)
        features = self.backbone(x)

        # Compute logits for each hierarchical level
        family_logits = self.family_head(features)
        genus_logits = self.genus_head(features)
        species_logits = self.species_head(features)

        return {
            "family": family_logits,
            "genus": genus_logits,
            "species": species_logits,
        }
