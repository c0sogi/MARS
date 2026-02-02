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
    When p=1, it acts as Average Pooling.
    When p -> infinity, it acts as Max Pooling.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        # Clamp to avoid NaN gradients with pow
        x = x.clamp(min=self.eps)
        # Apply Global Average Pooling on x^p
        # kernel_size matches the spatial dimensions (H, W)
        return F.avg_pool2d(x.pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class ArcMarginProduct(nn.Module):
    """
    ArcFace head implementation (Additive Angular Margin Loss).
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        # Class centers (Weights)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin

        # Precompute constants for margin logic
        self.update_margin_constants()

    def update_margin_constants(self):
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def set_margin(self, m):
        """
        Updates the margin value dynamically for curriculum learning.
        """
        self.m = m
        self.update_margin_constants()

    def forward(self, input, label=None):
        # --------------------------- cos(theta) ---------------------------
        # Normalize input features and weights
        # cosine = x . W / (|x| * |W|)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # If no label is provided (Inference), return scaled cosine similarities
        # or just raw cosine similarities. We return raw cosine here,
        # as ranking order is preserved regardless of 's'.
        if label is None:
            return cosine

        # --------------------------- cos(theta + m) ---------------------------
        # sin(theta) = sqrt(1 - cos(theta)^2)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Handle numerical stability / boundary conditions
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- Apply Margin ---------------------------
        # Convert label to one-hot implicitly by selecting indices
        # We only apply the margin penalty to the ground truth class

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # output = (one_hot * phi) + ((1 - one_hot) * cosine)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Scale the logits
        output *= self.s

        return output


class HotelRecognitionModel(nn.Module):
    """
    EfficientNet-B0 + GeM + ArcFace for Hotel Identification.
    """

    def __init__(
        self,
        n_classes=Config.NUM_CLASSES,
        model_name=Config.BACKBONE_NAME,
        embedding_dim=Config.EMBEDDING_DIM,
        margin=Config.MARGIN,
        scale=Config.SCALE,
        pretrained=True,
    ):
        super(HotelRecognitionModel, self).__init__()

        # 1. Backbone: EfficientNet-B0
        # num_classes=0 and global_pool='' returns the feature maps (B, C, H, W)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Get feature dimension (1280 for EfficientNet-B0)
        in_features = self.backbone.num_features

        # 2. Pooling: Generalized Mean Pooling
        self.pooling = GeM()

        # 3. Neck: Projection to Embedding Dimension
        # BN -> FC -> BN -> PReLU is a common block for metric learning
        self.neck = nn.Sequential(
            nn.Linear(in_features, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.PReLU(),  # Learnable activation
        )

        # 4. Head: ArcFace
        self.head = ArcMarginProduct(
            in_features=embedding_dim, out_features=n_classes, s=scale, m=margin
        )

    def forward(self, x, label=None):
        """
        Forward pass.
        Args:
            x: Input images (B, 3, H, W)
            label: Ground truth labels (B,). If None, returns cosine similarities.
        """
        # Extract features (B, C, H, W)
        features = self.backbone(x)

        # Pool (B, C, 1, 1)
        pooled = self.pooling(features)

        # Flatten (B, C)
        flattened = pooled.view(pooled.size(0), -1)

        # Project to embedding space (B, Emb_Dim)
        embeddings = self.neck(flattened)

        # Compute logits/similarities via ArcFace head
        logits = self.head(embeddings, label)

        return logits

    def extract_features(self, x):
        """
        Extracts normalized embeddings for inference/retrieval.
        """
        features = self.backbone(x)
        pooled = self.pooling(features)
        flattened = pooled.view(pooled.size(0), -1)
        embeddings = self.neck(flattened)
        return F.normalize(embeddings)

    def update_margin(self, m):
        """
        Updates the ArcFace margin (for curriculum learning).
        """
        self.head.set_margin(m)
