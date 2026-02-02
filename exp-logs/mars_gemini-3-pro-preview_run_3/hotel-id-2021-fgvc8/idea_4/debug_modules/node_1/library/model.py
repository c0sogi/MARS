import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes (1/N * sum(x^p))^(1/p).
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Clamp to avoid numerical instability with pow
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

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # Normalize features and weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # Calculate sin(theta)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

        # Calculate cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        # --------------------------- Apply Margin ---------------------------
        # Add margin only to the ground truth class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Scale the result
        output *= self.s

        return output


class HotelRecognitionModel(nn.Module):
    """
    Main model class for Hotel Identification.
    Integrates EfficientNet backbone, GeM pooling, BN Neck, and ArcFace Head.
    """

    def __init__(
        self,
        n_classes=Config.NUM_CLASSES,
        model_name=Config.BACKBONE_NAME,
        embedding_dim=Config.EMBEDDING_DIM,
        pretrained=Config.PRETRAINED,
        s=Config.ARCFACE_SCALE,
        m=Config.ARCFACE_MARGIN,
        easy_margin=Config.ARCFACE_EASY_MARGIN,
        ls_eps=Config.ARCFACE_LS_EPS,
    ):
        super(HotelRecognitionModel, self).__init__()

        # 1. Backbone
        # Create model but remove classifier (num_classes=0) to act as feature extractor
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )

        # Determine the number of output channels from the backbone
        # For EfficientNet-B0, this is typically 1280
        if hasattr(self.backbone, "num_features"):
            backbone_out = self.backbone.num_features
        else:
            # Fallback for common efficientnet
            backbone_out = 1280

        # 2. Pooling
        self.pooling = GeM()

        # 3. Neck (Linear Projection + Batch Normalization)
        # Project backbone features to the desired embedding dimension
        self.fc = nn.Linear(backbone_out, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)

        # Initialize Neck weights
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

        # 4. Head (ArcFace)
        self.arcface = ArcMarginProduct(
            in_features=embedding_dim,
            out_features=n_classes,
            s=s,
            m=m,
            easy_margin=easy_margin,
            ls_eps=ls_eps,
        )

    def forward(self, images, labels=None):
        """
        Forward pass.
        Args:
            images: Input images (B, C, H, W)
            labels: Target labels (B). Required for training.
        Returns:
            If labels are provided: ArcFace logits (B, Num_Classes)
            If labels are None: Embeddings (B, Embedding_Dim)
        """
        # Feature extraction
        features = self.backbone.forward_features(images)  # Output: (B, C, H, W)

        # Pooling
        features = self.pooling(features)  # Output: (B, C, 1, 1)
        features = features.flatten(1)  # Output: (B, C)

        # Neck
        features = self.fc(features)  # Output: (B, Emb_Dim)
        features = self.bn(features)  # Output: (B, Emb_Dim)

        # Training Mode: Return Logits
        if labels is not None:
            logits = self.arcface(features, labels)
            return logits

        # Inference Mode: Return Embeddings
        return features
