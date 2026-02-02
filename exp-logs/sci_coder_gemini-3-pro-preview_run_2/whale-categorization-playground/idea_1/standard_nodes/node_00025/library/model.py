import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling.
    Cite solution_lesson_node_00020
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )


class ArcMarginProduct(nn.Module):
    """
    ArcFace Head.
    Cite solution_lesson_node_00008
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

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class WhaleModel(nn.Module):
    """
    Backbone (EfficientNet) + GeM + ArcFace Head.
    Cite solution_lesson_node_00013, solution_lesson_node_00020
    """

    def __init__(self, num_classes):
        super(WhaleModel, self).__init__()

        # Backbone: EfficientNet-B3
        # num_classes=0 removes the head, global_pool='' keeps spatial dims
        self.backbone = timm.create_model(
            Config.BACKBONE, pretrained=Config.PRETRAINED, num_classes=0, global_pool=""
        )

        # Get feature dimension
        dummy_input = torch.randn(1, 3, 256, 256)
        features = self.backbone(dummy_input)
        in_features = features.shape[1]

        # Pooling
        self.pooling = GeM()

        # Neck (BN -> FC -> BN)
        self.bn1 = nn.BatchNorm1d(in_features)
        self.fc = nn.Linear(in_features, Config.EMBEDDING_DIM)
        self.bn2 = nn.BatchNorm1d(Config.EMBEDDING_DIM)

        # ArcFace Head
        self.arcface = ArcMarginProduct(
            Config.EMBEDDING_DIM, num_classes, s=Config.ARC_S, m=Config.ARC_M
        )

    def forward(self, x, labels=None):
        # Feature Extraction
        x = self.backbone(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)

        # Neck
        x = self.bn1(x)
        x = self.fc(x)
        x = self.bn2(x)

        # If labels are provided, return ArcFace logits (Training)
        if labels is not None:
            return self.arcface(x, labels)

        # Otherwise return Embeddings (Inference)
        return F.normalize(x, p=2, dim=1)


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss function.
    Minimizes distance for positive pairs (label=1) and maximizes distance
    for negative pairs (label=0) up to a margin.
    """

    def __init__(self, margin=Config.MARGIN):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """
        Args:
            output1 (torch.Tensor): Embeddings for first images.
            output2 (torch.Tensor): Embeddings for second images.
            label (torch.Tensor): 1.0 for same class, 0.0 for different class.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Calculate Euclidean distance
        euclidean_distance = F.pairwise_distance(output1, output2)

        # Formula: Y * D^2 + (1 - Y) * max(0, margin - D)^2
        # Note: dataset.py yields label=1 for same, label=0 for different.

        loss_contrastive = torch.mean(
            label * torch.pow(euclidean_distance, 2)
            + (1 - label)
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss_contrastive
