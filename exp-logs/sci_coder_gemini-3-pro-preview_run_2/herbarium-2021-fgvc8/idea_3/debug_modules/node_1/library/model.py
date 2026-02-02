import torch
import torch.nn as nn
import timm
from library.config import Config


class HierarchicalConvNeXt(nn.Module):
    """
    Hierarchical ConvNeXt model for plant classification.

    This architecture uses a ConvNeXt-Tiny backbone to extract features,
    followed by three parallel linear heads to predict:
    1. Species (Fine-grained, Main Task)
    2. Family (Coarse-grained, Auxiliary Task)
    3. Order (Coarse-grained, Auxiliary Task)

    This design supports the Decoupled Representation and Classifier Learning strategy.
    """

    def __init__(
        self,
        num_species=Config.NUM_CLASSES,
        num_families=None,
        num_orders=None,
        backbone_name=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
    ):
        """
        Args:
            num_species (int): Number of species classes (target).
            num_families (int): Number of family classes (auxiliary).
            num_orders (int): Number of order classes (auxiliary).
            backbone_name (str): Name of the timm backbone model.
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super(HierarchicalConvNeXt, self).__init__()

        if num_families is None or num_orders is None:
            raise ValueError(
                "num_families and num_orders must be provided to initialize the hierarchical heads."
            )

        # Create Backbone
        # num_classes=0 removes the top classification layer and returns the global pooled features
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )

        # Determine the number of input features for the heads
        if hasattr(self.backbone, "num_features"):
            self.n_features = self.backbone.num_features
        else:
            # Fallback mechanism if num_features is not explicitly exposed
            # For ConvNeXt and most timm models, num_features is available.
            # If not, we can usually infer it from the last layer's input features.
            # Creating a dummy input is another robust way, but usually unnecessary for standard models.
            self.n_features = self.backbone.head.in_features

        # Define Classification Heads
        self.species_head = nn.Linear(self.n_features, num_species)
        self.family_head = nn.Linear(self.n_features, num_families)
        self.order_head = nn.Linear(self.n_features, num_orders)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input batch of images, shape (B, C, H, W).

        Returns:
            dict: A dictionary containing logits for 'species', 'family', and 'order'.
        """
        # Extract features from the backbone
        # Shape: (Batch_Size, n_features)
        features = self.backbone(x)

        # Pass features through each head
        species_logits = self.species_head(features)
        family_logits = self.family_head(features)
        order_logits = self.order_head(features)

        return {
            "species": species_logits,
            "family": family_logits,
            "order": order_logits,
        }

    def freeze_backbone(self, freeze=True):
        """
        Freezes or unfreezes the backbone parameters.
        Useful for Stage 2 (Classifier Re-balancing) where we only want to train the head.

        Args:
            freeze (bool): If True, sets requires_grad=False for backbone parameters.
        """
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

    def freeze_auxiliary_heads(self, freeze=True):
        """
        Freezes or unfreezes the auxiliary heads (Family and Order).
        Useful for Stage 2 where we focus on the Species head.

        Args:
            freeze (bool): If True, sets requires_grad=False for family and order heads.
        """
        for param in self.family_head.parameters():
            param.requires_grad = not freeze

        for param in self.order_head.parameters():
            param.requires_grad = not freeze
