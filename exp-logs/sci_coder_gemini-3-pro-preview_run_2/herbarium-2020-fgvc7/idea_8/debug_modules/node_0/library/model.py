import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math


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
        # x: (B, C, H, W)
        # Output: (B, C, 1, 1)
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


class CosineLinear(nn.Module):
    """
    Cosine Linear layer (Cosine Classifier) for handling long-tail distributions.
    Normalizes weights and inputs to place them on a hypersphere.
    """

    def __init__(self, in_features, out_features, scale=30.0):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale

        # Weight parameter (out_features, in_features)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # Normalize input features
        x_norm = F.normalize(input, p=2, dim=1)
        # Normalize weights
        w_norm = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity
        cosine = F.linear(x_norm, w_norm)

        # Scale the output
        return self.scale * cosine


class CascadedEfficientNet(nn.Module):
    """
    Cascaded Hierarchical EfficientNet-B3.

    Structure:
    1. Backbone (EfficientNet-B3) -> GeM Pooling
    2. Family Head: Features -> Embedding -> Logits
    3. Genus Head: Concat(Features, Family Embedding) -> Embedding -> Logits
    4. Species Head: Concat(Features, Genus Embedding) -> Cosine Classifier -> Logits
    """

    def __init__(
        self,
        num_families,
        num_genera,
        num_species,
        backbone_name="efficientnet_b3",
        pretrained=True,
        embed_dim=512,
    ):
        super(CascadedEfficientNet, self).__init__()

        # Load Backbone
        # num_classes=0 removes the default classifier
        # global_pool='' removes the default pooling, keeping spatial features
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Get feature dimension (1536 for EfficientNet-B3)
        self.feat_dim = self.backbone.num_features

        # Pooling Layer
        self.pool = GeM()

        # Embedding Dimension for intermediate heads
        self.emb_dim = embed_dim

        # --- Family Head ---
        # Projects backbone features to Family Embedding
        self.family_embed = nn.Sequential(
            nn.Linear(self.feat_dim, self.emb_dim),
            nn.BatchNorm1d(self.emb_dim),
            nn.SiLU(),  # Swish activation
            nn.Dropout(p=0.2),
        )
        # Classifies Family
        self.family_classifier = nn.Linear(self.emb_dim, num_families)

        # --- Genus Head ---
        # Input: Backbone Features + Family Embedding
        self.genus_embed = nn.Sequential(
            nn.Linear(self.feat_dim + self.emb_dim, self.emb_dim),
            nn.BatchNorm1d(self.emb_dim),
            nn.SiLU(),
            nn.Dropout(p=0.2),
        )
        # Classifies Genus
        self.genus_classifier = nn.Linear(self.emb_dim, num_genera)

        # --- Species Head ---
        # Input: Backbone Features + Genus Embedding
        # Uses CosineLinear for the final classification
        self.species_input_dim = self.feat_dim + self.emb_dim
        self.species_classifier = CosineLinear(self.species_input_dim, num_species)

    def forward(self, x):
        # 1. Backbone Feature Extraction
        x = self.backbone(x)  # (B, 1536, H, W)
        x = self.pool(x)  # (B, 1536, 1, 1)
        x = x.flatten(1)  # (B, 1536)

        # 2. Family Prediction
        fam_emb = self.family_embed(x)  # (B, 512)
        fam_logits = self.family_classifier(fam_emb)  # (B, num_families)

        # 3. Genus Prediction (Conditioned on Family)
        # Concatenate backbone features with Family embedding
        genus_input = torch.cat([x, fam_emb], dim=1)  # (B, 1536 + 512)
        genus_emb = self.genus_embed(genus_input)  # (B, 512)
        genus_logits = self.genus_classifier(genus_emb)  # (B, num_genera)

        # 4. Species Prediction (Conditioned on Genus)
        # Concatenate backbone features with Genus embedding
        species_input = torch.cat([x, genus_emb], dim=1)  # (B, 1536 + 512)
        species_logits = self.species_classifier(species_input)  # (B, num_species)

        return species_logits, genus_logits, fam_logits
