import torch
import torch.nn as nn
import timm
from library.config import Config


class HierarchicalEfficientNet(nn.Module):
    """
    Hierarchical EfficientNetV2-Small model for plant classification.

    This architecture uses a shared backbone to extract features, which are then
    fed into three separate classification heads corresponding to the taxonomic
    hierarchy: Family, Genus, and Species.
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        pretrained=Config.PRETRAINED,
        num_classes_species=Config.NUM_CLASSES_SPECIES,
        num_classes_genus=Config.NUM_CLASSES_GENUS,
        num_classes_family=Config.NUM_CLASSES_FAMILY,
        dropout_rate=Config.DROPOUT_RATE,
        drop_path_rate=Config.DROP_PATH_RATE,
    ):
        """
        Args:
            model_name (str): Name of the timm model to use as backbone.
            pretrained (bool): Whether to load pretrained ImageNet weights.
            num_classes_species (int): Number of species classes (primary task).
            num_classes_genus (int): Number of genus classes (auxiliary task).
            num_classes_family (int): Number of family classes (auxiliary task).
            dropout_rate (float): Dropout probability before the classification heads.
            drop_path_rate (float): Stochastic depth rate for the backbone.
        """
        super(HierarchicalEfficientNet, self).__init__()

        # Initialize backbone
        # num_classes=0 removes the default FC layer
        # global_pool='' removes the default pooling, keeping spatial features (B, C, H, W)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            drop_path_rate=drop_path_rate,
        )

        # Get the number of features output by the backbone
        self.num_features = self.backbone.num_features

        # Global Average Pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=dropout_rate)

        # Multi-Task Heads
        # 1. Species Head (Fine-grained, Primary)
        self.species_head = nn.Linear(self.num_features, num_classes_species)

        # 2. Genus Head (Coarse-grained, Auxiliary)
        self.genus_head = nn.Linear(self.num_features, num_classes_genus)

        # 3. Family Head (Coarse-grained, Auxiliary)
        self.family_head = nn.Linear(self.num_features, num_classes_family)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, Channels, Height, Width).

        Returns:
            dict: A dictionary containing logits for each head:
                  {
                      'species': torch.Tensor,
                      'genus': torch.Tensor,
                      'family': torch.Tensor
                  }
        """
        # Extract features from backbone
        # Output shape: (B, num_features, H_grid, W_grid)
        features = self.backbone(x)

        # Apply Global Average Pooling
        # Output shape: (B, num_features, 1, 1)
        pooled = self.global_pool(features)

        # Flatten
        # Output shape: (B, num_features)
        flattened = pooled.flatten(1)

        # Apply Dropout
        x = self.dropout(flattened)

        # Pass through classification heads
        species_logits = self.species_head(x)
        genus_logits = self.genus_head(x)
        family_logits = self.family_head(x)

        return {
            "species": species_logits,
            "genus": genus_logits,
            "family": family_logits,
        }
