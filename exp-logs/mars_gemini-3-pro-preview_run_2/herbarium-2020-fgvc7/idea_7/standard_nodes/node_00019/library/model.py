import torch
import torch.nn as nn
import timm
from library.config import Config
from library.custom_layers import GeM, CosineClassifier
from library.taxonomy import TaxonomyMapper


class HierarchicalEfficientNet(nn.Module):
    """
    Hierarchical EfficientNet-B3 model for plant species classification.

    Features:
    - Backbone: EfficientNet-B3 (Noisy Student pretrained)
    - Pooling: Generalized Mean Pooling (GeM)
    - Species Head: Cosine Similarity Classifier (for long-tail robustness)
    - Auxiliary Heads: Standard Linear layers for Genus and Family prediction
    """

    def __init__(self, config: Config, mapper: TaxonomyMapper):
        """
        Args:
            config (Config): Configuration object containing model settings.
            mapper (TaxonomyMapper): Taxonomy object containing class counts for hierarchy.
        """
        super(HierarchicalEfficientNet, self).__init__()
        self.config = config

        # 1. Backbone: EfficientNet-B3
        # We set num_classes=0 and global_pool='' to get the raw spatial feature maps
        # (Batch, Channels, Height, Width) instead of pooled vectors.
        self.backbone = timm.create_model(
            config.BACKBONE, pretrained=True, num_classes=0, global_pool=""
        )

        # Determine input feature dimension (EfficientNet-B3 usually has 1536 channels)
        if hasattr(self.backbone, "num_features"):
            self.in_features = self.backbone.num_features
        else:
            self.in_features = config.EMBEDDING_DIM

        # 2. Pooling: Generalized Mean Pooling
        # Learnable parameter p allows the model to interpolate between Max and Avg pooling
        self.gem_pooling = GeM(p=3.0)

        # 3. Dropout for regularization
        self.dropout = nn.Dropout(p=0.3)

        # 4. Hierarchical Classification Heads

        # Species Head: Cosine Classifier
        # Normalizes features and weights to hypersphere, helping with rare classes
        self.species_head = CosineClassifier(
            in_features=self.in_features, out_features=mapper.num_classes, scale=30.0
        )

        # Genus Head: Standard Linear Classifier (Auxiliary Task)
        self.genus_head = nn.Linear(
            in_features=self.in_features, out_features=mapper.num_genera
        )

        # Family Head: Standard Linear Classifier (Auxiliary Task)
        self.family_head = nn.Linear(
            in_features=self.in_features, out_features=mapper.num_families
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input images of shape (Batch, 3, Height, Width).

        Returns:
            tuple: (species_logits, genus_logits, family_logits)
        """
        # Extract spatial features from backbone
        # Shape: (B, C, H, W)
        features = self.backbone(x)

        # Apply GeM Pooling
        # Shape: (B, C, 1, 1)
        pooled = self.gem_pooling(features)

        # Flatten
        # Shape: (B, C)
        flattened = pooled.flatten(1)

        # Apply Dropout
        embeddings = self.dropout(flattened)

        # Compute logits for each hierarchical level
        species_logits = self.species_head(embeddings)
        genus_logits = self.genus_head(embeddings)
        family_logits = self.family_head(embeddings)

        return species_logits, genus_logits, family_logits
