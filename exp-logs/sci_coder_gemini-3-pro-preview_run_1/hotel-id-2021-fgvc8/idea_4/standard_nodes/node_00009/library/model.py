import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes (1/N * sum(x^p))^(1/p) with a learnable parameter p.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x: (B, C, H, W)
        # Clamp for numerical stability
        x = x.clamp(min=eps).pow(p)
        # Average pooling over spatial dimensions
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        # Root p
        return x.pow(1.0 / p)

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"
        )


class ArcMarginProduct(nn.Module):
    """
    ArcFace (Additive Angular Margin) classification head.
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        # Learnable class centers (weights)
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

        # sin(theta) = sqrt(1 - cos(theta)^2)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

        # phi = cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Handle numerical stability when theta + m > pi
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        # Create a mask for the ground truth labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # --------------------------- Final Logits ---------------------------
        # Apply margin only to the ground truth class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Scale the logits
        output *= self.s

        return output


class HotelIdModel(nn.Module):
    """
    GeM-Augmented EfficientNet Retrieval System.
    Integrates EfficientNet backbone, GeM pooling, and ArcFace head.
    """

    def __init__(
        self,
        n_classes=Config.NUM_CLASSES,
        backbone_name=Config.BACKBONE,
        embedding_dim=Config.EMBEDDING_DIM,
        pretrained=Config.PRETRAINED,
        gem_p=Config.GEM_P,
        margin=Config.MARGIN,
        scale=Config.SCALE,
    ):
        super(HotelIdModel, self).__init__()

        # 1. Backbone: EfficientNet
        # num_classes=0 and global_pool='' returns the spatial feature map (B, C, H, W)
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Determine the number of input features from the backbone
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.backbone(dummy_input)
            in_features = features.shape[1]

        # 2. Pooling: GeM
        self.gem = GeM(p=gem_p)

        # 3. Neck: Projection to Embedding Dimension
        # BN is used to normalize features before the ArcFace head
        self.neck = nn.Sequential(
            nn.Linear(in_features, embedding_dim), nn.BatchNorm1d(embedding_dim)
        )

        # 4. Head: ArcFace
        self.head = ArcMarginProduct(
            in_features=embedding_dim, out_features=n_classes, s=scale, m=margin
        )

    def forward(self, x, labels=None):
        """
        Forward pass.
        Args:
            x: Input images (B, 3, H, W)
            labels: Ground truth labels (B,). If None, returns embeddings.
        """
        # Feature Extraction
        x = self.backbone(x)  # (B, C, H, W)

        # Pooling
        x = self.gem(x)  # (B, C, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to (B, C)

        # Embedding Projection
        embedding = self.neck(x)  # (B, 512)

        if labels is not None:
            # Training: Return logits with ArcFace margin
            return self.head(embedding, labels)
        else:
            # Inference: Return normalized embeddings
            return F.normalize(embedding, p=2, dim=1)

    def extract_features(self, x):
        """
        Helper for inference to explicitly get embeddings.
        """
        return self.forward(x, labels=None)
