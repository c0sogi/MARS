import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from library.config import Config


class ArcMarginProduct(nn.Module):
    """
    ArcFace output layer.
    Cite Lesson 00008: Classification-Based Margin Losses.
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
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # -------------torch.where(out_i = {x_i if condition_i else y_i}) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class WhaleArcFaceModel(nn.Module):
    """
    Model with EfficientNet Backbone and ArcFace Head.
    Cite Lesson 00013: Capacity-Resolution Swap (EfficientNet-B3).
    """

    def __init__(self, num_classes, backbone_name=Config.BACKBONE, pretrained=True):
        super(WhaleArcFaceModel, self).__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )
        self.in_features = self.backbone.num_features

        self.bn1 = nn.BatchNorm1d(self.in_features)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.in_features, Config.EMBEDDING_DIM)
        self.bn2 = nn.BatchNorm1d(Config.EMBEDDING_DIM)

        self.arcface = ArcMarginProduct(
            Config.EMBEDDING_DIM, num_classes, s=Config.ARCFACE_S, m=Config.ARCFACE_M
        )

    def forward(self, x, labels=None, extract_embeddings=False):
        features = self.backbone(x)
        features = self.bn1(features)
        features = self.dropout(features)
        embeddings = self.fc1(features)
        embeddings = self.bn2(embeddings)

        # L2 normalize embeddings for inference
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)

        if extract_embeddings:
            return embeddings_norm

        if labels is not None:
            return self.arcface(embeddings, labels)
        return embeddings_norm
