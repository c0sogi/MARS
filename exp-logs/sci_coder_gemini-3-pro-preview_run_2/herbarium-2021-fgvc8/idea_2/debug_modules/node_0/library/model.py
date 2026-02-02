import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class ArcMarginProduct(nn.Module):
    r"""
    Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
    """

    def __init__(
        self,
        in_features,
        out_features,
        s=30.0,
        m=0.50,
        easy_margin=False,
        ls_eps=0.0,
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # cosine = input . weight / (|input| * |weight|)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # If no label is provided (inference), just return scaled cosine similarities
        if label is None:
            return cosine * self.s

        # --------------------------- cos(theta + m) ---------------------------
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        # Create one_hot mask for the target labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Optional: Label smoothing on the mask itself
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        # --------------------------- output ---------------------------
        # Apply margin only to the target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class HierarchicalEfficientNet(nn.Module):
    """
    Hierarchical EfficientNet with ArcFace for Species and Linear heads for Family/Order.
    """

    def __init__(
        self,
        backbone_name=Config.BACKBONE,
        pretrained=True,
        num_classes=Config.NUM_CLASSES,
        num_families=1,  # Should be provided by the training script
        num_orders=1,
        embedding_dim=Config.EMBEDDING_DIM,
        dropout=Config.DROPOUT,
        s=Config.ARCFACE_SCALE,
        m=Config.ARCFACE_MARGIN,
    ):
        super(HierarchicalEfficientNet, self).__init__()

        # 1. Backbone
        # num_classes=0 means we get the pooled feature vector
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )
        self.num_features = self.backbone.num_features

        # 2. Embedding / Projection Layer
        # Projects backbone features to a lower dimensional embedding space for ArcFace
        self.embedding_layer = nn.Sequential(
            nn.Linear(self.num_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.Dropout(dropout),
        )

        # 3. Heads
        # Species Head: Metric Learning (ArcFace)
        self.species_head = ArcMarginProduct(embedding_dim, num_classes, s=s, m=m)

        # Auxiliary Heads: Standard Classification
        self.family_head = nn.Linear(embedding_dim, num_families)
        self.order_head = nn.Linear(embedding_dim, num_orders)

    def forward(self, x, species_label=None):
        """
        Args:
            x: Input images [B, C, H, W]
            species_label: Ground truth species labels [B].
                           Required during training for ArcFace margin calculation.
                           Can be None during inference.
        Returns:
            species_logits, family_logits, order_logits
        """
        # Extract features from backbone
        features = self.backbone(x)

        # Project to embedding space
        embeddings = self.embedding_layer(features)

        # Forward pass through heads
        species_logits = self.species_head(embeddings, species_label)
        family_logits = self.family_head(embeddings)
        order_logits = self.order_head(embeddings)

        return species_logits, family_logits, order_logits
