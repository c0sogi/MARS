import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library import config


class ArcMarginProduct(nn.Module):
    r"""
    Implement of large margin cosine distance:
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
    ):
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
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # input is (Batch, Dim), weight is (Classes, Dim)
        # Normalize features and weights to project onto hypersphere
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # If no label is provided (Inference/Validation), return scaled cosine similarities
        if label is None:
            return cosine * self.s

        # --------------------------- Margin Penalty ---------------------------
        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- Apply to Target Class ---------------------------
        # Create one-hot encoding for the labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Apply margin only to the ground truth class
        # output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = torch.where(one_hot.bool(), phi, cosine)

        # Scale the logits
        output *= self.s

        return output


class WhaleDenseNet(nn.Module):
    def __init__(
        self,
        num_classes=config.NUM_CLASSES,
        embedding_dim=config.EMBEDDING_DIM,
        pretrained=config.PRETRAINED,
    ):
        """
        Whale Identification Model based on DenseNet121.

        Structure:
        1. Backbone: DenseNet121 (Pretrained)
        2. Pooling: Global Average Pooling
        3. Neck: Linear -> BatchNorm (Projection Head)
        4. Head: ArcFace (Additive Angular Margin)
        """
        super(WhaleDenseNet, self).__init__()

        # 1. Backbone
        weights = "DEFAULT" if pretrained else None
        self.backbone = models.densenet121(weights=weights)

        # Get input features for the classifier (1024 for DenseNet121)
        in_features = self.backbone.classifier.in_features

        # Remove original classifier to save memory/parameters
        del self.backbone.classifier

        self.use_neck = config.USE_NECK

        # 2. Neck (Projection Head)
        if self.use_neck:
            self.neck = nn.Sequential(
                nn.Linear(in_features, embedding_dim), nn.BatchNorm1d(embedding_dim)
            )
            self.head_in_features = embedding_dim
        else:
            self.neck = nn.Identity()
            self.head_in_features = in_features

        # 3. Head (ArcFace)
        self.arcface = ArcMarginProduct(
            in_features=self.head_in_features,
            out_features=num_classes,
            s=config.SCALE,
            m=config.MARGIN,
        )

    def forward(self, x, labels=None):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input images (Batch, 3, H, W)
            labels (torch.Tensor, optional): Ground truth labels for ArcFace margin.
                                             If None, returns raw cosine logits.
        """
        # Feature Extraction
        features = self.backbone.features(x)

        # DenseNet Post-Processing: BN -> ReLU -> GAP
        # Note: backbone.features ends with a BatchNorm in DenseNet implementation
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)

        # Projection Neck
        embedding = self.neck(out)

        # ArcFace Head
        output = self.arcface(embedding, labels)

        return output
