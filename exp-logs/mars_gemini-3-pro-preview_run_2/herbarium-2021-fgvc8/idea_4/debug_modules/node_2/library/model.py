import torch
import torch.nn as nn
import timm
from library.config import Config


class HierarchicalConvNeXt(nn.Module):
    """
    Hierarchical Multi-Task Classification Model using ConvNeXt-Tiny backbone.
    Predicts Species, Family, and Order simultaneously.
    """

    def __init__(self, num_families, num_orders, pretrained=Config.PRETRAINED):
        """
        Args:
            num_families (int): Number of family classes.
            num_orders (int): Number of order classes.
            pretrained (bool): Whether to load pretrained backbone weights.
        """
        super().__init__()

        # 1. Backbone: ConvNeXt-Tiny
        # num_classes=0 removes the default classifier
        # global_pool='avg' ensures the output is a flattened feature vector (B, num_features)
        self.backbone = timm.create_model(
            Config.BACKBONE, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # Get the feature dimension (usually 768 for convnext_tiny)
        self.num_features = self.backbone.num_features
        hidden_dim = Config.HEAD_HIDDEN_DIM

        # 2. Multi-Task Heads
        # We use a bottleneck architecture for the heads as per Config
        self.species_head = self._build_head(
            self.num_features, Config.NUM_CLASSES, hidden_dim
        )
        self.family_head = self._build_head(self.num_features, num_families, hidden_dim)
        self.order_head = self._build_head(self.num_features, num_orders, hidden_dim)

    def _build_head(self, in_dim, out_dim, hidden_dim):
        """
        Constructs a classification head with a hidden projection layer.
        Structure: Linear -> BatchNorm -> ReLU -> Dropout -> Linear
        """
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W)
        Returns:
            tuple: (species_logits, family_logits, order_logits)
        """
        # Extract features from backbone
        features = self.backbone(x)

        # Pass through each head
        species_logits = self.species_head(features)
        family_logits = self.family_head(features)
        order_logits = self.order_head(features)

        return species_logits, family_logits, order_logits

    def freeze_backbone(self):
        """
        Freezes the backbone parameters.
        Used for Stage 2 (Classifier Re-balancing).
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """
        Unfreezes the backbone parameters.
        Used for Stage 1 (Representation Learning).
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
