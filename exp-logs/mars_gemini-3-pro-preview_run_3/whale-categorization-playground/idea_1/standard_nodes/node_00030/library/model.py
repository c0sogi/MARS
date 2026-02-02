import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class ArcMarginProduct(nn.Module):
    """
    ArcFace output layer.
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

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.easy_margin = easy_margin

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
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i}) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class WhaleEmbeddingNet(nn.Module):
    """
    EfficientNet-B2 backbone with ArcFace Head.
    Cite solution_lesson_node_00010: EfficientNet-B2 with Attention.
    Cite solution_lesson_node_00008: ArcFace.
    """

    def __init__(self, num_classes=None, embedding_dim=None):
        super(WhaleEmbeddingNet, self).__init__()

        if embedding_dim is None:
            embedding_dim = Config.EMBEDDING_DIM

        # Load pre-trained EfficientNet-B2
        self.backbone = models.efficientnet_b2(weights="DEFAULT")

        # Remove the classifier
        self.backbone.classifier = nn.Identity()

        # EfficientNet-B2 last channel size is 1408
        num_ftrs = 1408

        # Pooling
        self.pooling = nn.AdaptiveAvgPool2d(1)

        # Neck: BN -> Dropout -> Linear -> BN (Cite solution_lesson_node_00029)
        self.bn1 = nn.BatchNorm1d(num_ftrs)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_ftrs, embedding_dim)
        self.bn2 = nn.BatchNorm1d(embedding_dim)

        # ArcFace Head
        if num_classes is not None:
            self.arcface = ArcMarginProduct(
                embedding_dim, num_classes, s=Config.ARC_S, m=Config.ARC_M
            )
        else:
            self.arcface = None

    def forward(self, x, labels=None):
        # Backbone features
        x = self.backbone.features(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)

        # Neck
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc(x)
        features = self.bn2(x)

        # If training with labels, return ArcFace logits
        if labels is not None and self.arcface is not None:
            return self.arcface(features, labels)

        # Otherwise return normalized embeddings for inference
        return F.normalize(features, p=2, dim=1)

    def get_embedding(self, x):
        return self.forward(x, labels=None)
