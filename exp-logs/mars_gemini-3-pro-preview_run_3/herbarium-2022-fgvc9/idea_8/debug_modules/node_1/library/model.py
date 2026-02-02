import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the p-th power of the input, averages it, and takes the p-th root.
    p is a learnable parameter.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        # We use average pooling on x^p, then take the (1/p)-th root.
        # clamp(min=eps) ensures numerical stability.
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)


class HierarchicalEfficientNet(nn.Module):
    """
    Hierarchical EfficientNetV2-B0 model with GeM pooling and multi-task heads.
    """

    def __init__(self, num_species, num_genera, num_families, pretrained=True):
        """
        Args:
            num_species (int): Number of species classes (primary task).
            num_genera (int): Number of genus classes (auxiliary task).
            num_families (int): Number of family classes (auxiliary task).
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super(HierarchicalEfficientNet, self).__init__()

        # Load EfficientNetV2-B0 backbone.
        # num_classes=0 removes the fully connected layer.
        # global_pool='' removes the default pooling layer, returning spatial features.
        self.backbone = timm.create_model(
            "tf_efficientnetv2_b0", pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Retrieve the number of output features from the backbone (typically 1280 for B0)
        self.num_features = self.backbone.num_features

        # Generalized Mean Pooling
        self.pooling = GeM()

        # Multi-task Classification Heads
        # Primary Head
        self.head_species = nn.Linear(self.num_features, num_species)
        # Auxiliary Heads
        self.head_genus = nn.Linear(self.num_features, num_genera)
        self.head_family = nn.Linear(self.num_features, num_families)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W).

        Returns:
            dict: Dictionary containing logits for 'species', 'genus', and 'family'.
        """
        # Extract spatial features from backbone
        # Shape: (B, C, H_feat, W_feat)
        features = self.backbone(x)

        # Apply GeM pooling
        # Shape: (B, C, 1, 1)
        pooled = self.pooling(features)

        # Flatten features
        # Shape: (B, C)
        embeddings = pooled.flatten(1)

        # Compute logits for each taxonomic level
        logits_species = self.head_species(embeddings)
        logits_genus = self.head_genus(embeddings)
        logits_family = self.head_family(embeddings)

        return {
            "species": logits_species,
            "genus": logits_genus,
            "family": logits_family,
        }
