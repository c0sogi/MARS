import torch
import torch.nn as nn
import timm
from library.utils import Config


class HierarchicalConvNeXt(nn.Module):
    """
    Hierarchical Multi-Task Learning Model based on ConvNeXt-Base.

    This model uses a shared backbone to extract features and three parallel
    classification heads to predict taxonomic levels: Family, Genus, and Species.
    """

    def __init__(self, pretrained=True):
        """
        Args:
            pretrained (bool): Whether to load pre-trained ImageNet-21k weights.
        """
        super(HierarchicalConvNeXt, self).__init__()

        # Initialize the backbone
        # num_classes=0 returns the pooled feature vector (after final norm)
        self.backbone = timm.create_model(
            Config.MODEL_NAME, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # Retrieve the embedding dimension of the backbone
        n_features = self.backbone.num_features

        # Define parallel classification heads
        self.head_species = nn.Linear(n_features, Config.NUM_CLASSES_SPECIES)
        self.head_genus = nn.Linear(n_features, Config.NUM_CLASSES_GENUS)
        self.head_family = nn.Linear(n_features, Config.NUM_CLASSES_FAMILY)

        # Initialize weights for heads (optional, PyTorch defaults are usually fine)
        self._init_weights(self.head_species)
        self._init_weights(self.head_genus)
        self._init_weights(self.head_family)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W)

        Returns:
            dict: A dictionary containing logits for 'species', 'genus', and 'family'.
        """
        # Extract features using the backbone
        features = self.backbone(x)

        # Pass features through each head
        logits_species = self.head_species(features)
        logits_genus = self.head_genus(features)
        logits_family = self.head_family(features)

        return {
            "species": logits_species,
            "genus": logits_genus,
            "family": logits_family,
        }
