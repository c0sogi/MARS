import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B3_Weights


class HierarchicalEfficientNet(nn.Module):
    """
    EfficientNet-B3 based model with Dual-Pooling and Hierarchical Multi-Task Heads.

    Architecture:
    - Backbone: EfficientNet-B3 (pretrained)
    - Pooling: Concatenation of Global Average Pooling and Global Max Pooling
    - Heads:
        1. Species Head (Target)
        2. Genus Head (Auxiliary)
        3. Family Head (Auxiliary)
    """

    def __init__(self, num_species, num_genus, num_family, pretrained=True):
        """
        Args:
            num_species (int): Number of species classes (output of species head).
            num_genus (int): Number of genus classes (output of genus head).
            num_family (int): Number of family classes (output of family head).
            pretrained (bool): If True, loads ImageNet pretrained weights for the backbone.
        """
        super(HierarchicalEfficientNet, self).__init__()

        # Load backbone with appropriate weights
        if pretrained:
            weights = EfficientNet_B3_Weights.DEFAULT
        else:
            weights = None

        self.backbone = models.efficientnet_b3(weights=weights)

        # Determine the number of output features from the backbone.
        # EfficientNet-B3 typically has 1536 channels in the final feature map.
        # We check the input features of the original classifier to be sure.
        self.num_features = self.backbone.classifier[1].in_features

        # Remove the original classifier to save parameters and avoid confusion,
        # though we will only use .features() in forward.
        self.backbone.classifier = nn.Identity()

        # Dual Pooling Layers
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        # Input dimension for the heads is doubled due to concatenation of Avg and Max pooling
        in_features = self.num_features * 2

        # Hierarchical Classification Heads
        self.fc_species = nn.Linear(in_features, num_species)
        self.fc_genus = nn.Linear(in_features, num_genus)
        self.fc_family = nn.Linear(in_features, num_family)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (Tensor): Input images of shape (N, 3, H, W).

        Returns:
            dict: A dictionary containing logits for each taxonomic level:
                  {
                      'species': Tensor (N, num_species),
                      'genus': Tensor (N, num_genus),
                      'family': Tensor (N, num_family)
                  }
        """
        # Extract features from the backbone (N, C, H, W)
        x = self.backbone.features(x)

        # Apply Dual Pooling
        x_avg = self.avg_pool(x).flatten(1)
        x_max = self.max_pool(x).flatten(1)

        # Concatenate features (N, C*2)
        x_cat = torch.cat([x_avg, x_max], dim=1)

        # Pass through hierarchical heads
        out_species = self.fc_species(x_cat)
        out_genus = self.fc_genus(x_cat)
        out_family = self.fc_family(x_cat)

        return {"species": out_species, "genus": out_genus, "family": out_family}


def get_model(num_species, num_genus, num_family, pretrained=True):
    """
    Factory function to create the HierarchicalEfficientNet model.
    """
    return HierarchicalEfficientNet(num_species, num_genus, num_family, pretrained)
