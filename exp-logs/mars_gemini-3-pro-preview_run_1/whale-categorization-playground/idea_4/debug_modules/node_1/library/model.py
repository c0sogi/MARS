import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class ArcMarginProduct(nn.Module):
    """
    Implementation of ArcFace (Additive Angular Margin Loss).
    Computes cos(theta + m) for the target class to enforce larger margins.
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        # Weights for the class centers (Prototypes)
        # Shape: (out_features, in_features)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # --------------------------- Cosine Similarity ---------------------------
        # Normalize input features and weights to lie on the hypersphere
        # input: (Batch, in_features)
        # weight: (Out_features, in_features)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # If no label is provided (Inference), return the raw scaled cosine similarities.
        # These logits represent the similarity to each class center.
        if label is None:
            return cosine * self.s

        # --------------------------- Angular Margin (Training) ---------------------------
        # cos(theta) = cosine
        # sin(theta) = sqrt(1 - cos^2(theta))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

        # phi = cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Strictly enforce margin only when theta + m < pi
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- Apply to Targets ---------------------------
        # Create one-hot encoding for the targets
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Calculate output: use phi (margin) for target class, cosine (raw) for others
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Scale the result
        output *= self.s

        return output


class WhaleConvNeXt(nn.Module):
    """
    Whale Species Predictor using ConvNeXt-Tiny and ArcFace.
    Structure: Backbone -> GAP -> BN (Neck) -> ArcFace (Head)
    """

    def __init__(self):
        super(WhaleConvNeXt, self).__init__()

        # 1. Backbone: ConvNeXt Tiny
        # We disable the built-in head and pooling to handle them manually.
        # drop_path_rate applies Stochastic Depth regularization.
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="",
            drop_path_rate=Config.DROP_PATH_RATE,
        )

        # Determine the number of input features for the head
        # Run a dummy forward pass to get shapes (ConvNeXt-Tiny is usually 768)
        with torch.no_grad():
            dummy = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE)
            features = self.backbone(dummy)
            # features shape: (1, C, H, W)
            self.num_features = features.shape[1]

        # 2. Neck: Batch Normalization
        # Normalizes the pooled features before the metric head.
        # This is critical for the stability of ArcFace training.
        self.neck = nn.BatchNorm1d(self.num_features)

        # 3. Head: ArcFace
        self.head = ArcMarginProduct(
            in_features=self.num_features,
            out_features=Config.NUM_CLASSES,
            s=Config.ARC_SCALE,
            m=Config.ARC_MARGIN,
        )

    def forward(self, x, labels=None):
        """
        Forward pass of the model.

        Args:
            x: Input images (B, 3, H, W)
            labels: Target labels (B,) or None.

        Returns:
            Tensor: Logits.
            - If labels are provided, returns logits with angular margin penalty (for Loss).
            - If labels are None, returns scaled cosine similarities (for Prediction).
        """
        # Feature Extraction
        x = self.backbone(x)  # (B, C, H, W)

        # Global Average Pooling (GAP)
        # Average over spatial dimensions (H, W)
        x = x.mean(dim=[-2, -1])  # (B, C)

        # Neck (BN)
        x = self.neck(x)

        # Head (ArcFace)
        logits = self.head(x, labels)

        return logits

    def extract_features(self, x):
        """
        Returns normalized embeddings for the input images.
        This bypasses the classification head, useful for embedding-based analysis.

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            Tensor: Normalized embeddings (B, C)
        """
        x = self.backbone(x)
        x = x.mean(dim=[-2, -1])
        x = self.neck(x)
        return F.normalize(x)
