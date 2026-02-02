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
        # x shape: (B, C, H, W)
        # Clamp to avoid NaN in power
        x = x.clamp(min=eps)
        # Average pooling on x^p
        x_pow = x.pow(p)
        # Global Average Pooling over (H, W)
        avg_x_pow = F.avg_pool2d(x_pow, (x.size(-2), x.size(-1)))
        # (Avg(x^p))^(1/p)
        return avg_x_pow.pow(1.0 / p)

    def __repr__(self):
        return f"GeM(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class SubCenterArcFaceHead(nn.Module):
    """
    ArcFace Head with Sub-Centers (K centers per class).
    """

    def __init__(self, in_features, num_classes, k=3, s=30.0, m=0.50):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.k = k
        self.s = s
        self.m = m

        # Weights shape: (num_classes * k, in_features)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes * k, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precompute margin constants
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels=None):
        """
        Args:
            embeddings: (Batch, Embedding_Dim)
            labels: (Batch,) - Ground truth labels. If None, returns raw logits scaled by s.
        """
        # Normalize features and weights
        norm_embeddings = F.normalize(embeddings, dim=1)
        norm_weight = F.normalize(self.weight, dim=1)

        # Compute Cosine Similarity: (Batch, Num_Classes * K)
        cosine = F.linear(norm_embeddings, norm_weight)

        # Reshape to (Batch, Num_Classes, K)
        cosine = cosine.view(-1, self.num_classes, self.k)

        # Max over sub-centers -> (Batch, Num_Classes)
        # We take the best matching sub-center for each class
        cosine, _ = torch.max(cosine, dim=2)

        if labels is None:
            # Inference mode or feature extraction validation
            return cosine * self.s

        # --- Training Mode with Margin ---

        # Get cosine values for the ground truth classes
        # labels shape: (B,) -> (B, 1)
        labels = labels.view(-1, 1).long()
        cosine_target = cosine.gather(1, labels)

        # Calculate cos(theta + m)
        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        sin_theta = torch.sqrt(1.0 - torch.pow(cosine_target, 2).clamp(0, 1))
        cosine_target_margin = cosine_target * self.cos_m - sin_theta * self.sin_m

        # Stability check (keep gradients well-behaved)
        # If cos_theta > cos(pi - m), apply margin normally.
        # Otherwise, use a penalty approximation (cos_theta - mm).
        cosine_target_margin = torch.where(
            cosine_target > self.th, cosine_target_margin, cosine_target - self.mm
        )

        # Create one-hot encoding to update only the target class logits
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels, 1.0)

        # Apply margin to targets, keep others as is
        logits = cosine * (1.0 - one_hot) + cosine_target_margin * one_hot

        # Scale
        logits *= self.s

        return logits


class HotelRecognitionModel(nn.Module):
    """
    Main model class for Hotel ID Recognition.
    Combines Backbone + GeM + Neck + SubCenterArcFaceHead.
    """

    def __init__(
        self,
        backbone_name,
        num_classes=Config.NUM_CLASSES,
        embedding_dim=Config.EMBEDDING_DIM,
        pretrained=True,
    ):
        super(HotelRecognitionModel, self).__init__()

        # 1. Backbone
        # Create model without classifier and without global pooling
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Determine input features for the neck
        if hasattr(self.backbone, "num_features"):
            in_features = self.backbone.num_features
        else:
            # Fallback for some models, compute by passing dummy input
            with torch.no_grad():
                dummy = torch.randn(1, 3, 256, 256)
                out = self.backbone(dummy)
                in_features = out.shape[1]

        # 2. Pooling
        self.pooling = GeM()

        # 3. Neck (Projection to Embedding Space)
        # Linear -> BN -> PReLU is a standard block for ReID/FaceID
        self.neck = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.PReLU(),
        )

        # 4. Head (Sub-Center ArcFace)
        self.head = SubCenterArcFaceHead(
            in_features=embedding_dim,
            num_classes=num_classes,
            k=Config.SUB_CENTER_K,
            s=Config.ARCFACE_SCALE,
            m=Config.ARCFACE_MARGIN,
        )

    def forward(self, x, labels=None):
        """
        Forward pass.

        Args:
            x: Input images (B, C, H, W)
            labels: Ground truth labels (B,). Required for training loss calculation.

        Returns:
            If labels is NOT None (Training):
                logits: (B, Num_Classes) - Scaled logits with margin applied to targets.
            If labels IS None (Inference):
                embeddings: (B, Embedding_Dim) - Projected feature embeddings.
        """
        # Feature Extraction
        features = self.backbone(x)  # (B, C, H, W)

        # Pooling
        features = self.pooling(features)  # (B, C, 1, 1)
        features = features.view(features.size(0), -1)  # Flatten to (B, C)

        # Projection
        embeddings = self.neck(features)  # (B, Embedding_Dim)

        if labels is not None:
            # Training: Pass through head to get logits for loss
            logits = self.head(embeddings, labels)
            return logits
        else:
            # Inference: Return embeddings for retrieval
            return embeddings
