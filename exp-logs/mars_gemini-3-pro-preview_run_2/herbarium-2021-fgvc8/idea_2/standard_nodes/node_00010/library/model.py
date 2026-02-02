import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class HierarchicalModel(nn.Module):
    """
    Hierarchical Model (ResNet18) with Standard Linear Head for Species and Linear heads for Family/Order.
    """

    def __init__(
        self,
        backbone_name=Config.BACKBONE,
        pretrained=True,
        num_classes=Config.NUM_CLASSES,
        num_families=1,
        num_orders=1,
        embedding_dim=Config.EMBEDDING_DIM,
        dropout=Config.DROPOUT,
    ):
        super(HierarchicalModel, self).__init__()

        # 1. Backbone
        # num_classes=0 means we get the pooled feature vector
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )
        self.num_features = self.backbone.num_features

        # 2. Embedding / Projection Layer
        self.embedding_layer = nn.Sequential(
            nn.Linear(self.num_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.Dropout(dropout),
        )

        # 3. Heads
        # Species Head: Standard Classification
        self.species_head = nn.Linear(embedding_dim, num_classes)

        # Auxiliary Heads: Standard Classification
        self.family_head = nn.Linear(embedding_dim, num_families)
        self.order_head = nn.Linear(embedding_dim, num_orders)

    def forward(self, x, species_label=None):
        """
        Args:
            x: Input images [B, C, H, W]
            species_label: Ignored (kept for compatibility with engine).
        Returns:
            species_logits, family_logits, order_logits
        """
        # Extract features from backbone
        features = self.backbone(x)

        # Project to embedding space
        embeddings = self.embedding_layer(features)

        # Forward pass through heads
        species_logits = self.species_head(embeddings)
        family_logits = self.family_head(embeddings)
        order_logits = self.order_head(embeddings)

        return species_logits, family_logits, order_logits
