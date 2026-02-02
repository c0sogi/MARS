import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes (Avg(x^p))^(1/p).
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


class ArcMarginProduct(nn.Module):
    """
    ArcFace (Additive Angular Margin Loss) Head.
    """

    def __init__(
        self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # --------------------------- cosine ---------------------------
        # input: (B, Embedding_Size)
        # weight: (Num_Classes, Embedding_Size)
        # cosine: (B, Num_Classes)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # If no label is provided (inference), return scaled cosine similarities
        if label is None:
            return cosine * self.s

        # --------------------------- cos(theta + m) ---------------------------
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Handle numerical stability when theta + m > pi
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        # Apply margin only to the ground truth class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class HotelEfficientNet(nn.Module):
    """
    Main Model Architecture:
    EfficientNet-B0 -> GeM Pooling -> Embedding Neck -> ArcFace Head
    """

    def __init__(
        self,
        num_classes=Config.NUM_CLASSES,
        backbone_name=Config.BACKBONE,
        pretrained=True,
    ):
        super(HotelEfficientNet, self).__init__()

        # 1. Backbone
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained)

        # Get input features for the classifier (usually 1280 for B0)
        in_features = self.backbone.num_features

        # Remove default head and pooling
        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()

        # 2. Pooling
        self.pooling = GeM()

        # 3. Neck (Embedding Layer)
        self.embedding_size = Config.EMBEDDING_SIZE
        self.neck = nn.Sequential(
            nn.Linear(in_features, self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
            nn.Dropout(p=Config.DROPOUT),
        )

        # 4. Head (ArcFace)
        self.arcface = ArcMarginProduct(
            in_features=self.embedding_size,
            out_features=num_classes,
            s=Config.ARC_SCALE,
            m=Config.ARC_MARGIN,
        )

    def forward(self, images, labels=None):
        """
        Forward pass.
        Args:
            images: (B, C, H, W) input images
            labels: (B,) ground truth labels (optional)
        Returns:
            logits: (B, Num_Classes)
        """
        # Extract features: (B, C, H, W)
        features = self.backbone.forward_features(images)

        # GeM Pooling: (B, C, H, W) -> (B, C, 1, 1)
        pooled_features = self.pooling(features)

        # Flatten: (B, C, 1, 1) -> (B, C)
        flattened_features = pooled_features.view(pooled_features.size(0), -1)

        # Embedding: (B, C) -> (B, Embedding_Size)
        embeddings = self.neck(flattened_features)

        # ArcFace Head: (B, Embedding_Size) -> (B, Num_Classes)
        logits = self.arcface(embeddings, labels)

        return logits

    def extract_features(self, images):
        """
        Helper for inference to get embeddings directly.
        """
        features = self.backbone.forward_features(images)
        pooled_features = self.pooling(features)
        flattened_features = pooled_features.view(pooled_features.size(0), -1)
        embeddings = self.neck(flattened_features)
        return embeddings
