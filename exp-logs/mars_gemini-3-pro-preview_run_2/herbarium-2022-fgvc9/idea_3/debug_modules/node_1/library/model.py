import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from library.config import Config


class ArcFaceLayer(nn.Module):
    """
    ArcFace (Additive Angular Margin Loss) Layer.

    Reference: Deng et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition".
    """

    def __init__(self, in_features, out_features, scale=30.0, margin=0.50):
        super(ArcFaceLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin

        # Learnable weights (class centers)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precompute cos(m) and sin(m)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

        # Threshold for numerical stability
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label=None):
        # 1. Normalize Features and Weights
        # input: [batch_size, in_features]
        # weight: [out_features, in_features]
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # If inference or no label provided, return scaled cosine logits
        if label is None:
            return cosine * self.scale

        # 2. Add Margin Penalty (only for ground truth classes)
        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        # Handle numerical stability issues where theta + m > pi
        # If cos(theta) > th, use phi. Else use cosine - mm (Taylor approximation fallback)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 3. Convert labels to one-hot to apply margin only to target class
        # logits = s * (cos(theta + m)) for target
        # logits = s * cos(theta) for others
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output


class HierarchicalMetricNet(nn.Module):
    """
    Hierarchical Metric Learning Network.

    Backbone: EfficientNet-B4 (Unfrozen)
    Heads:
        1. Species: ArcFace Head (Metric Learning)
        2. Genus: Linear Head (Auxiliary)
        3. Family: Linear Head (Auxiliary)
    """

    def __init__(self, num_species, num_genera, num_families):
        super(HierarchicalMetricNet, self).__init__()

        # 1. Backbone
        # Load pretrained EfficientNet, remove classification head (num_classes=0)
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            drop_rate=Config.DROP_RATE,
            drop_path_rate=Config.DROP_PATH_RATE,
            global_pool="avg",
        )

        # Get feature dimension (e.g., 1792 for B4)
        self.backbone_dim = self.backbone.num_features

        # 2. Embedding Projection
        # Projects high-dim backbone features to a compact embedding space
        self.embedding_dim = Config.EMBEDDING_DIM
        self.projection = nn.Sequential(
            nn.Linear(self.backbone_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.PReLU(),
            nn.Dropout(p=Config.DROP_RATE),
        )

        # 3. Heads
        # Species Head: ArcFace
        self.species_head = ArcFaceLayer(
            in_features=self.embedding_dim,
            out_features=num_species,
            scale=Config.ARCFACE_SCALE,
            margin=Config.ARCFACE_MARGIN,
        )

        # Auxiliary Heads: Standard Linear
        # We attach these to the embedding as well to enforce structure in the latent space
        self.genus_head = nn.Linear(self.embedding_dim, num_genera)
        self.family_head = nn.Linear(self.embedding_dim, num_families)

    def forward(self, x, species_label=None):
        """
        Args:
            x (torch.Tensor): Input images [B, C, H, W]
            species_label (torch.Tensor, optional): Ground truth species indices [B].
                                                    Required for ArcFace training.

        Returns:
            dict: Dictionary containing logits for 'species', 'genus', 'family',
                  and the raw 'embedding'.
        """
        # Feature Extraction
        features = self.backbone(x)

        # Projection to Embedding Space
        embedding = self.projection(features)

        # Head Forward Passes
        # Note: ArcFace needs labels during training to add margin
        species_logits = self.species_head(embedding, species_label)
        genus_logits = self.genus_head(embedding)
        family_logits = self.family_head(embedding)

        return {
            "species": species_logits,
            "genus": genus_logits,
            "family": family_logits,
            "embedding": embedding,
        }

    def get_embedding(self, x):
        """Helper for inference to just get embeddings."""
        features = self.backbone(x)
        return self.projection(features)
