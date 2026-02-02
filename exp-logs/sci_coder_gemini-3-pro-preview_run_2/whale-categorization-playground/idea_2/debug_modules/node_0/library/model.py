import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
from library.config import MODEL_NAME, EMBEDDING_DIM, NUM_CLASSES, MARGIN, SCALE


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes (1/N * sum(x^p))^(1/p).
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


class ArcMarginProduct(nn.Module):
    """
    ArcFace (Additive Angular Margin Loss) Layer.
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        # Weights for the classification layer (centers)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # Normalize input features and weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # Calculate sin(theta)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

        # Calculate cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Keep phi only if cos(theta) > cos(pi - m) to ensure monotonicity
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- Convert label to one-hot ---------------------------
        # Create one-hot encoding of labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # --------------------------- Apply Margin ---------------------------
        # Apply phi to target class, cosine to others
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Scale the logits
        output *= self.s

        return output


class WhaleArcFaceModel(nn.Module):
    """
    Main model class for Humpback Whale Identification.
    Backbone: EfficientNet-B0
    Pooling: GeM
    Head: ArcFace
    """

    def __init__(
        self,
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        embedding_dim=EMBEDDING_DIM,
        pretrained=True,
    ):
        super(WhaleArcFaceModel, self).__init__()

        # 1. Backbone
        # num_classes=0 and global_pool="" ensures we get feature maps (B, C, H, W)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Determine input features dimension dynamically
        dummy_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            features = self.backbone(dummy_input)
            in_features = features.shape[1]

        # 2. Pooling Layer
        self.pooling = GeM()

        # 3. Embedding Neck (BN -> FC -> BN)
        self.bn1 = nn.BatchNorm1d(in_features)
        self.fc = nn.Linear(in_features, embedding_dim)
        self.bn2 = nn.BatchNorm1d(embedding_dim)

        # 4. ArcFace Head (Training only)
        self.arcface = ArcMarginProduct(
            in_features=embedding_dim, out_features=num_classes, s=SCALE, m=MARGIN
        )

    def forward(self, images, labels=None):
        """
        Args:
            images: Input images (B, 3, H, W)
            labels: Target labels (B,) - Optional

        Returns:
            If labels are provided (Training): ArcFace logits (B, Num_Classes)
            If labels are None (Inference): Embeddings (B, Embedding_Dim)
        """
        # Feature Extraction
        features = self.backbone(images)  # (B, C, H, W)

        # Pooling
        features = self.pooling(features)  # (B, C, 1, 1)
        features = features.flatten(1)  # (B, C)

        # Embedding Projection
        features = self.bn1(features)
        embeddings = self.fc(features)
        embeddings = self.bn2(embeddings)

        # Training Mode: Return Logits with Margin
        if labels is not None:
            logits = self.arcface(embeddings, labels)
            return logits

        # Inference Mode: Return Embeddings
        return embeddings
