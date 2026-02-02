import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library import config


class ArcMarginProduct(nn.Module):
    """
    Implement of large margin cosine distance:
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # --------------------------- cos(theta) & phi(theta + m) ---------------------------
        # L2 Normalize input and weights
        # input: (batch, in_features)
        # weight: (out_features, in_features)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # If inference (no label), return scaled cosine similarity
        if label is None:
            return cosine * self.s

        # --------------------------- Training ---------------------------
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # For numerical stability and to handle theta + m > pi
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Create one-hot encoding for labels
        # label shape: (batch_size)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Apply margin only to the ground truth class
        # output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = (one_hot * phi) + (cosine * (1.0 - one_hot))

        # Scale the logits
        output *= self.s

        return output


class HierarchicalResNet(nn.Module):
    """
    Hierarchical Multi-Task Network.
    Backbone: ResNet-50
    Head 1: Species Classification (ArcFace)
    Head 2: Genus Classification (CrossEntropy/Linear)
    """

    def __init__(
        self, num_species, num_genus, backbone_name="resnet50", pretrained=True
    ):
        super(HierarchicalResNet, self).__init__()

        # 1. Backbone
        if backbone_name == "resnet50":
            # Use weights="DEFAULT" for best available pretrained weights
            weights = "DEFAULT" if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
        else:
            # Fallback
            weights = "DEFAULT" if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = 2048

        # Remove original fully connected layer
        self.backbone.fc = nn.Identity()

        # 2. Embedding Layer (Bottleneck)
        # Standard ArcFace practice: BN -> Dropout -> FC -> BN
        self.embedding_dim = config.EMBEDDING_DIM

        self.bn1 = nn.BatchNorm1d(in_features)
        self.dropout = nn.Dropout(p=0.4)
        self.fc_embedding = nn.Linear(in_features, self.embedding_dim)
        self.bn2 = nn.BatchNorm1d(self.embedding_dim)

        # 3. Heads
        # Primary Head: Species (ArcFace)
        self.species_head = ArcMarginProduct(
            in_features=self.embedding_dim,
            out_features=num_species,
            s=config.ARCFACE_SCALE,
            m=config.ARCFACE_MARGIN,
        )

        # Auxiliary Head: Genus (Standard Linear)
        self.genus_head = nn.Linear(self.embedding_dim, num_genus)

    def forward(self, x, species_label=None):
        """
        Args:
            x (torch.Tensor): Input images (Batch, 3, H, W)
            species_label (torch.Tensor, optional): Ground truth species labels.
                                                    Required for ArcFace training.
        Returns:
            species_logits: Output of ArcFace head.
            genus_logits: Output of Genus head.
        """
        # Extract features from backbone
        # ResNet50 (fc=Identity) returns (Batch, 2048)
        features = self.backbone(x)

        # Embedding block
        features = self.bn1(features)
        features = self.dropout(features)
        embedding = self.fc_embedding(features)
        embedding = self.bn2(embedding)

        # Heads
        species_logits = self.species_head(embedding, species_label)
        genus_logits = self.genus_head(embedding)

        return species_logits, genus_logits
