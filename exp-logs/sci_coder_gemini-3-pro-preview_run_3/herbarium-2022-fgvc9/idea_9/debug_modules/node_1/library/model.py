import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x shape: (B, C, H, W)
        # clamp min to avoid NaN in pow
        # Average pool over H, W
        x_pool = F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1)))
        # Pow 1/p and flatten to (B, C)
        return x_pool.pow(1.0 / p).flatten(1)


class CascadingPlantModel(nn.Module):
    """
    Cascading Hierarchical EfficientNetV2-B0.

    Structure:
    - Backbone: EfficientNetV2-B0
    - Pooling: GeM
    - Heads:
        1. Family Head: Features -> Family Logits
        2. Genus Head: Concat(Features, Proj(Family Logits)) -> Genus Logits
        3. Species Head: Concat(Features, Proj(Genus Logits)) -> Species Logits
    """

    def __init__(
        self,
        num_species,
        num_genera,
        num_families,
        backbone_name="tf_efficientnetv2_b0",
        pretrained=True,
        proj_dim=512,
    ):
        super(CascadingPlantModel, self).__init__()

        # 1. Backbone
        # Create model without classifier and global pooling to get spatial features
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Get feature dimension (1280 for efficientnetv2_b0)
        self.num_features = self.backbone.num_features

        # 2. Pooling
        self.gem = GeM()

        # 3. Family Branch
        self.fc_family = nn.Linear(self.num_features, num_families)

        # Projection for Family Logits -> Embedding
        self.proj_family = nn.Sequential(
            nn.Linear(num_families, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
        )

        # 4. Genus Branch
        # Input: Image Features + Projected Family Logits
        self.fc_genus = nn.Linear(self.num_features + proj_dim, num_genera)

        # Projection for Genus Logits -> Embedding
        self.proj_genus = nn.Sequential(
            nn.Linear(num_genera, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
        )

        # 5. Species Branch
        # Input: Image Features + Projected Genus Logits
        self.fc_species = nn.Linear(self.num_features + proj_dim, num_species)

    def forward(self, x):
        # Extract features (B, C, H, W)
        features = self.backbone(x)

        # Pooling (B, C)
        features = self.gem(features)

        # --- Family Prediction ---
        family_logits = self.fc_family(features)

        # --- Genus Prediction ---
        # Project family logits and concatenate
        fam_emb = self.proj_family(family_logits)
        genus_input = torch.cat([features, fam_emb], dim=1)
        genus_logits = self.fc_genus(genus_input)

        # --- Species Prediction ---
        # Project genus logits and concatenate
        # Note: We concatenate features with genus info.
        # The genus info implicitly contains family info from previous step.
        gen_emb = self.proj_genus(genus_logits)
        species_input = torch.cat([features, gen_emb], dim=1)
        species_logits = self.fc_species(species_input)

        return species_logits, genus_logits, family_logits
