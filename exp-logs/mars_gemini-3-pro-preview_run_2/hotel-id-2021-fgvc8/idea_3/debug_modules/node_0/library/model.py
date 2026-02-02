import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes (1/N * sum(x^p))^(1/p).
    When p=1, acts as Average Pooling.
    When p->inf, acts as Max Pooling.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x: (B, C, H, W)
        # Clamp to avoid NaN in pow()
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )


class SubCenterArcFace(nn.Module):
    """
    Sub-Center ArcFace Head.
    Maintains K centers per class to handle intra-class variance.
    """

    def __init__(self, in_features, num_classes, k=3, s=30.0, m=0.50):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.k = k
        self.s = s
        self.m = m

        # Weights shape: (NumClasses, K, EmbeddingDim)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, k, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precompute constants for ArcFace margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, features, labels):
        # features: (Batch, EmbeddingDim)
        # labels: (Batch)

        # 1. Normalize Input Features
        x = F.normalize(features, dim=1)

        # 2. Normalize Weights
        # Reshape to (NumClasses * K, EmbeddingDim) for efficient matrix multiplication
        w = F.normalize(self.weight, dim=2)
        w = w.view(-1, self.in_features)

        # 3. Compute Cosine Similarity
        # (Batch, EmbDim) @ (EmbDim, NumClasses*K) -> (Batch, NumClasses*K)
        cosine = F.linear(x, w)

        # 4. Reshape and Select Best Sub-Center
        # (Batch, NumClasses, K)
        cosine = cosine.view(-1, self.num_classes, self.k)
        # Take max over K dimension -> (Batch, NumClasses)
        cosine, _ = torch.max(cosine, dim=2)

        # 5. Apply ArcFace Margin to Ground Truth
        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        # Handle numerical stability / monotonicity
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Create One-Hot encoding for labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Combine: Margin for GT, original cosine for others
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # 6. Scale
        output *= self.s

        return output


class HotelModel(nn.Module):
    """
    Main model architecture for Hotel Identification.
    Backbone: EfficientNet-B4
    Pooling: GeM
    Neck: Linear + BN
    Head: Sub-Center ArcFace
    """

    def __init__(self, num_classes=Config.NUM_CLASSES):
        super(HotelModel, self).__init__()

        # Backbone
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="",  # Return spatial features (B, C, H, W)
        )

        # Feature dimension from backbone
        in_features = self.backbone.num_features

        # Pooling Layer
        if Config.USE_GEM_POOLING:
            self.pooling = GeM()
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)

        # Neck (Projection to Embedding Dimension)
        self.neck = nn.Sequential(
            nn.Linear(in_features, Config.EMBEDDING_DIM, bias=False),
            nn.BatchNorm1d(Config.EMBEDDING_DIM),
        )

        # Metric Learning Head
        self.head = SubCenterArcFace(
            in_features=Config.EMBEDDING_DIM,
            num_classes=num_classes,
            k=Config.NUM_SUB_CENTERS,
            s=Config.SCALE,
            m=Config.MARGIN,
        )

    def forward(self, x, labels=None):
        """
        Forward pass.
        Args:
            x: Input images (B, C, H, W)
            labels: Ground truth labels (B). If None, returns embeddings.
        """
        # Feature Extraction
        x = self.backbone(x)  # (B, C, H, W)

        # Pooling
        x = self.pooling(x)  # (B, C, 1, 1)
        x = x.flatten(1)  # (B, C)

        # Embedding Projection
        embeddings = self.neck(x)  # (B, EmbDim)

        if labels is not None:
            # Training Mode: Compute ArcFace Logits
            logits = self.head(embeddings, labels)
            return logits
        else:
            # Inference Mode: Return Embeddings
            return embeddings
