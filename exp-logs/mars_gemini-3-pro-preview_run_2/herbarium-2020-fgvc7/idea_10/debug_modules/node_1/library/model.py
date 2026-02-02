import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes the generalized mean: f(X) = (1/N * sum(x^p))^(1/p)
    When p=1, it acts as Average Pooling.
    When p->infinity, it acts as Max Pooling.
    The parameter p is learnable.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        # Clamp to avoid numerical instability with power
        x = torch.clamp(x, min=self.eps)

        # Calculate spatial dimensions for averaging
        h, w = x.size(2), x.size(3)

        # Apply GeM formula
        # (Average(x^p))^(1/p)
        x_pow = x.pow(self.p)
        avg_x_pow = F.avg_pool2d(x_pow, (h, w))
        gem_out = avg_x_pow.pow(1.0 / self.p)

        return gem_out

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class CosineClassifier(nn.Module):
    """
    Cosine Similarity Classifier Head.
    Instead of dot product (w * x + b), it computes cosine similarity:
    (w . x) / (|w| * |x|) * scale

    This normalizes both weights and features to a hypersphere, helping with
    class imbalance by removing magnitude bias.
    """

    def __init__(self, in_features, out_features, scale=30.0, margin=0.0):
        super(CosineClassifier, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin

        # Weight matrix (out_features, in_features)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # Normalize features (input)
        x_norm = F.normalize(x, p=2, dim=1)

        # Normalize weights
        w_norm = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity
        cosine = F.linear(x_norm, w_norm)

        # Apply scaling
        logits = self.scale * cosine

        return logits


class HierarchicalEfficientNet(nn.Module):
    """
    EfficientNet-B3 with GeM pooling and Hierarchical Heads (Family, Genus, Species).
    The Species head uses a Cosine Classifier.
    """

    def __init__(self, n_families, n_genera, n_species, pretrained=True):
        super(HierarchicalEfficientNet, self).__init__()

        # 1. Backbone
        # Load EfficientNet-B3 using timm
        # features_only=False gives the full model, we strip classifier later.
        # num_classes=0 removes the top classifier layer in timm.
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",  # Disable default pooling to use GeM
        )

        # Get feature dimension (EfficientNet-B3 usually 1536)
        self.num_features = self.backbone.num_features

        # 2. Pooling
        self.pool = GeM(p=3.0)

        # 3. Heads
        # Standard Linear Heads for coarse levels
        self.fc_family = nn.Linear(self.num_features, n_families)
        self.fc_genus = nn.Linear(self.num_features, n_genera)

        # Cosine Classifier for the fine-grained species level
        self.fc_species = CosineClassifier(self.num_features, n_species)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # Feature Extraction
        # Shape: (B, C, H, W)
        features = self.backbone(x)

        # Pooling
        # Shape: (B, C, 1, 1)
        pooled_features = self.pool(features)

        # Flatten
        # Shape: (B, C)
        flattened_features = torch.flatten(pooled_features, 1)

        # Apply Dropout
        features_drop = self.dropout(flattened_features)

        # Heads
        # We use the same feature vector for all heads (multi-task learning)

        # Family Prediction
        logits_family = self.fc_family(features_drop)

        # Genus Prediction
        logits_genus = self.fc_genus(features_drop)

        # Species Prediction (Cosine)
        logits_species = self.fc_species(features_drop)

        return {
            "species": logits_species,
            "genus": logits_genus,
            "family": logits_family,
            "features": flattened_features,  # Return features if needed for embedding analysis
        }

    def get_params_groups(self):
        """
        Separate parameters for differential learning rates if needed.
        """
        backbone_params = list(self.backbone.parameters())
        head_params = (
            list(self.pool.parameters())
            + list(self.fc_family.parameters())
            + list(self.fc_genus.parameters())
            + list(self.fc_species.parameters())
        )
        return backbone_params, head_params
