import torch
import torch.nn as nn
import timm


class HierarchicalEfficientNet(nn.Module):
    """
    EfficientNet-B4 with Hierarchical Multi-Task Heads for Plant Classification.

    This model uses a shared EfficientNet-B4 backbone to extract visual features,
    which are then fed into three separate fully connected heads to predict
    Family, Genus, and Species simultaneously. This structure leverages
    taxonomic relationships to improve fine-grained species classification.
    """

    def __init__(
        self, num_species, num_genera, num_families, pretrained=True, dropout_p=0.3
    ):
        """
        Initialize the hierarchical model.

        Args:
            num_species (int): Number of unique species (target classes).
            num_genera (int): Number of unique genera (auxiliary classes).
            num_families (int): Number of unique families (auxiliary classes).
            pretrained (bool): Whether to load pretrained ImageNet weights for the backbone.
            dropout_p (float): Dropout probability applied to the feature vector.
        """
        super(HierarchicalEfficientNet, self).__init__()

        # Load EfficientNet-B4 backbone
        # 'tf_efficientnet_b4_ns' uses Noisy Student weights which often perform better
        # num_classes=0 removes the default classification layer
        # global_pool='avg' ensures the output is a pooled feature vector (Batch, Num_Features)
        self.backbone = timm.create_model(
            "tf_efficientnet_b4_ns",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        # Retrieve the number of input features for the classification heads
        # For EfficientNet-B4, this is typically 1792
        in_features = self.backbone.num_features

        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout_p)

        # Define the three independent classification heads
        self.species_head = nn.Linear(in_features, num_species)
        self.genus_head = nn.Linear(in_features, num_genera)
        self.family_head = nn.Linear(in_features, num_families)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input images of shape [Batch, 3, Height, Width].

        Returns:
            tuple: A tuple containing:
                - species_logits (torch.Tensor): [Batch, num_species]
                - genus_logits (torch.Tensor): [Batch, num_genera]
                - family_logits (torch.Tensor): [Batch, num_families]
        """
        # Extract features from the backbone
        features = self.backbone(x)

        # Apply dropout to shared features
        features = self.dropout(features)

        # Compute logits for each taxonomic level
        species_logits = self.species_head(features)
        genus_logits = self.genus_head(features)
        family_logits = self.family_head(features)

        return species_logits, genus_logits, family_logits
