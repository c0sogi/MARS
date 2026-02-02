import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from library.config import Config


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
        # x is expected to be (B, C, H, W)
        # Clamp min=eps to ensure numerical stability and validity for pow operation
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
    Implementation of Additive Angular Margin Loss (ArcFace).
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
        # input: (B, in_features) - Normalized embeddings
        # label: (B) - Ground truth labels

        # Cosine similarity: W * x
        # Both input and weight should be normalized
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Convert label to one-hot
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Apply margin only to the ground truth class
        # output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class WhaleModel(nn.Module):
    """
    Backbone model with GeM pooling and Projection Head.
    """

    def __init__(self, backbone_name=None, embedding_size=None, pretrained=True):
        super(WhaleModel, self).__init__()

        if backbone_name is None:
            backbone_name = Config.backbone
        if embedding_size is None:
            embedding_size = Config.embedding_size

        # 1. Backbone
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained)

        # Determine in_features and remove original head
        if hasattr(self.backbone, "num_features"):
            self.in_features = self.backbone.num_features
        else:
            # Fallback for models where num_features isn't directly exposed
            self.in_features = self.backbone.classifier.in_features

        # Remove the classifier and global pooling to get raw feature maps
        self.backbone.reset_classifier(0)

        # 2. Pooling
        self.pooling = GeM()

        # 3. Projection Head
        # Structure: BN -> Dropout -> Linear -> BN
        self.neck_bn = nn.BatchNorm1d(self.in_features)
        self.dropout = nn.Dropout(p=0.2)
        self.dense = nn.Linear(self.in_features, embedding_size)
        self.head_bn = nn.BatchNorm1d(embedding_size)

        # Weight Initialization for the head
        nn.init.kaiming_normal_(self.dense.weight)
        nn.init.constant_(self.dense.bias, 0)
        nn.init.constant_(self.head_bn.weight, 1)
        nn.init.constant_(self.head_bn.bias, 0)
        nn.init.constant_(self.neck_bn.weight, 1)
        nn.init.constant_(self.neck_bn.bias, 0)

    def forward(self, x):
        # Extract features from backbone
        # Shape: (B, C, H, W)
        features = self.backbone.forward_features(x)

        # Pooling
        # Shape: (B, C, 1, 1) -> (B, C)
        pooled = self.pooling(features)
        flattened = pooled.view(pooled.size(0), -1)

        # Projection Head
        x = self.neck_bn(flattened)
        x = self.dropout(x)
        x = self.dense(x)
        embeddings = self.head_bn(x)

        return embeddings
