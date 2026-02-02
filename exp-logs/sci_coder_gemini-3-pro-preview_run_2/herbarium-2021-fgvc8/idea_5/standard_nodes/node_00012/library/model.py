import torch
import torch.nn as nn
import timm
from library.config import Config
from library.utils import load_taxonomy_mapping


class HierarchicalConvNeXt(nn.Module):
    """
    Hierarchical Multi-Task Learning Model using ConvNeXt-Tiny backbone.
    Predicts Species, Family, and Order simultaneously.
    """

    def __init__(self, pretrained=Config.PRETRAINED):
        """
        Args:
            pretrained (bool): Whether to load pretrained weights for the backbone.
        """
        super().__init__()

        # ------------------------------------------------------------------
        # 1. Determine Output Dimensions for Heads
        # ------------------------------------------------------------------
        # Load taxonomy mapping to get counts for auxiliary tasks
        taxonomy_df = load_taxonomy_mapping(load_cached_data=True)

        # Species count is fixed in Config, others are derived from data
        self.num_species = Config.NUM_CLASSES
        self.num_families = taxonomy_df["family_id"].max() + 1
        self.num_orders = taxonomy_df["order_id"].max() + 1

        # ------------------------------------------------------------------
        # 2. Initialize Backbone
        # ------------------------------------------------------------------
        # Create ConvNeXt Tiny backbone
        # num_classes=0 removes the default classification head
        # global_pool='avg' ensures we get a feature vector (B, C)
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            drop_rate=Config.DROPOUT,
            drop_path_rate=Config.DROP_PATH_RATE,
        )

        # Get the feature dimension (usually 768 for convnext_tiny)
        self.in_features = self.backbone.num_features

        # ------------------------------------------------------------------
        # 3. Define Classification Heads
        # ------------------------------------------------------------------
        # Primary Head
        self.species_head = nn.Linear(self.in_features, self.num_species)

        # Auxiliary Heads (for hierarchical regularization)
        self.family_head = nn.Linear(self.in_features, self.num_families)
        self.order_head = nn.Linear(self.in_features, self.num_orders)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W)

        Returns:
            dict: Dictionary containing logits for 'species', 'family', and 'order'.
        """
        # Extract features from backbone
        features = self.backbone(x)

        # Pass features through classification heads
        species_logits = self.species_head(features)
        family_logits = self.family_head(features)
        order_logits = self.order_head(features)

        return {
            "species": species_logits,
            "family": family_logits,
            "order": order_logits,
        }

    def freeze_backbone(self):
        """
        Freezes the backbone parameters.
        Used during Stage 2 (Classifier Re-balancing).
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """
        Unfreezes the backbone parameters.
        Used during Stage 1 (Representation Learning) or fine-tuning.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
