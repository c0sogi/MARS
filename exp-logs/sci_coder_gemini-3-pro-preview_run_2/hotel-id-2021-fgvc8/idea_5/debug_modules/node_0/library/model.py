import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
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


class SubCenterArcFace(nn.Module):
    """
    Sub-Center ArcFace Head.
    Maintains K centers per class to handle multi-modal distributions.
    """

    def __init__(self, in_features, out_features, k=3, s=30.0, m=0.50):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.s = s
        self.m = m

        # Weight shape: (out_features * k, in_features)
        # We flatten the sub-centers into the first dimension for efficient computation
        self.weight = nn.Parameter(torch.FloatTensor(out_features * k, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Pre-compute constants for ArcFace margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (B, in_features)
            labels: (B,)
        Returns:
            logits: (B, out_features) with ArcFace margin applied to ground truth
        """
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, dim=1)
        weights = F.normalize(self.weight, dim=1)

        # Compute cosine similarity
        # (B, in_features) @ (in_features, out_features * k) -> (B, out_features * k)
        cosine = F.linear(embeddings, weights)

        # Reshape to (B, out_features, k) to separate sub-centers
        cosine = cosine.view(-1, self.out_features, self.k)

        # Take the maximum cosine similarity across the k sub-centers
        # This dynamically assigns the sample to the nearest sub-center
        cosine, _ = torch.max(cosine, dim=2)  # (B, out_features)

        # --- ArcFace Margin Logic ---
        # Create one-hot mask for ground truth labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Calculate phi = cos(theta + m)
        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        # Stability check: if cos(theta) > cos(pi - m), use phi. Else use cosine - mm.
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Apply margin only to ground truth classes
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Scale the logits
        output *= self.s

        return output


class HotelConvNeXt(nn.Module):
    """
    Main model class combining ConvNeXt backbone, GeM pooling, and SubCenter ArcFace head.
    """

    def __init__(
        self,
        backbone_name=Config.backbone,
        embedding_size=Config.embedding_size,
        num_classes=Config.num_classes,
        k_subcenters=Config.k_subcenters,
        margin=Config.margin,
        scale=Config.scale,
        pretrained=True,
    ):
        super().__init__()

        # Backbone: ConvNeXt Base
        # global_pool='' ensures we get spatial features (B, C, H, W) for GeM
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool=""
        )
        in_features = self.backbone.num_features

        # Pooling: Generalized Mean Pooling
        self.pool = GeM()

        # Neck: Projection to embedding dimension + BatchNorm
        self.neck = nn.Sequential(
            nn.Linear(in_features, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size),
        )

        # Head: Sub-Center ArcFace
        self.head = SubCenterArcFace(
            in_features=embedding_size,
            out_features=num_classes,
            k=k_subcenters,
            s=scale,
            m=margin,
        )

    def forward(self, x, labels=None):
        """
        Forward pass.
        Args:
            x: Input images (B, C, H, W)
            labels: Ground truth labels (B,) [Optional]
        Returns:
            If labels are provided (Training): Logits with margin (B, num_classes)
            If labels are None (Inference): Normalized embeddings (B, embedding_size)
        """
        # Feature extraction
        x = self.backbone(x)  # (B, C, H, W)
        x = self.pool(x)  # (B, C, 1, 1)
        x = x.flatten(1)  # (B, C)
        x = self.neck(x)  # (B, embedding_size)

        if labels is not None:
            # Training phase: Pass through head to compute logits with margin
            return self.head(x, labels)
        else:
            # Inference phase: Return L2-normalized embeddings
            return F.normalize(x, dim=1)
