import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class HierarchicalEfficientNet(nn.Module):
    """
    Hierarchical EfficientNet with Standard Linear Heads for Species, Family, and Order.
    Simplified from ArcFace to improve convergence on large-scale data (Cite solution_lesson_node_00008).
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
        # s and m removed as ArcFace is replaced
    ):
        super(HierarchicalEfficientNet, self).__init__()

        # 1. Backbone
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
        # Species Head: Standard Classification (Linear)
        self.species_head = nn.Linear(embedding_dim, num_classes)

        # Auxiliary Heads: Standard Classification
        self.family_head = nn.Linear(embedding_dim, num_families)
        self.order_head = nn.Linear(embedding_dim, num_orders)

    def forward(self, x):
        """
        Args:
            x: Input images [B, C, H, W]
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
