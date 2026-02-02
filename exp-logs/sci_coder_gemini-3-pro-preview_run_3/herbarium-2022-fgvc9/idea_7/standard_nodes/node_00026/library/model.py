import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the generalized mean of the spatial features, allowing the model
    to learn to focus on salient regions (high activations) rather than
    averaging everything (like Global Average Pooling).

    Formula: f(X) = (1/|X| * sum(x^p))^(1/p)
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter, initialized to 3
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Clamp to avoid NaN, power p, avg pool, power 1/p
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class HierarchicalEfficientNet(nn.Module):
    """
    Hierarchical EfficientNetV2 Model.

    Features:
    - Backbone: EfficientNetV2-Small (pretrained).
    - Pooling: GeM (Generalized Mean Pooling).
    - Heads: Three parallel heads for Species, Genus, and Family.
    """

    def __init__(self, pretrained=True):
        super(HierarchicalEfficientNet, self).__init__()

        # 1. Create Backbone
        # num_classes=0 removes the default linear head
        # global_pool='' ensures we get the spatial feature map (B, C, H, W)
        self.backbone = timm.create_model(
            Config.MODEL_NAME, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Get the number of output features from the backbone
        num_features = self.backbone.num_features

        # 2. Pooling Layer
        if Config.USE_GEM_POOLING:
            self.pooling = GeM()
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)

        # 3. Regularization
        self.dropout = nn.Dropout(p=Config.DROPOUT_RATE)

        # 4. Hierarchical Classification Heads
        # Primary Task
        self.head_species = nn.Linear(num_features, Config.NUM_CLASSES)

        # Auxiliary Tasks
        self.head_genus = nn.Linear(num_features, Config.NUM_GENERA)
        self.head_family = nn.Linear(num_features, Config.NUM_FAMILIES)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input images (B, C, H, W).

        Returns:
            dict: Dictionary containing logits for 'species', 'genus', and 'family'.
        """
        # Extract features from backbone: (B, C, H, W)
        features = self.backbone.forward_features(x)

        # Apply Pooling: (B, C, 1, 1) -> (B, C)
        pooled_features = self.pooling(features).flatten(1)

        # Apply Dropout
        pooled_features = self.dropout(pooled_features)

        # Compute Logits for each hierarchy level
        logits_species = self.head_species(pooled_features)
        logits_genus = self.head_genus(pooled_features)
        logits_family = self.head_family(pooled_features)

        return {
            "species": logits_species,
            "genus": logits_genus,
            "family": logits_family,
        }
