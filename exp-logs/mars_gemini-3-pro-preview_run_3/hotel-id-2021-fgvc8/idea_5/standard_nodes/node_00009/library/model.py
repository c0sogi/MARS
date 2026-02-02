import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes (1/N * sum(x^p))^(1/p) with a learnable p.
    This allows the model to learn to focus on salient regions of the image.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x shape: (Batch, Channels, Height, Width)
        # Clamp inputs to eps to ensure numerical stability (and handle negative values from Swish/SiLU)
        x = x.clamp(min=eps)
        # Raise to power p
        x_pow = x.pow(p)
        # Average pool over spatial dimensions
        avg_pool = F.avg_pool2d(x_pow, (x.size(-2), x.size(-1)))
        # Take the p-th root
        return avg_pool.pow(1.0 / p)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class ArcMarginProduct(nn.Module):
    """
    ArcFace Head.
    Implements Additive Angular Margin Loss.
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

        # Class centers (weights)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # Normalize input features and weights to lie on the hypersphere
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # Calculate sin(theta)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

        # Calculate phi = cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Handle boundary conditions
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- Apply Margin to Target Class ---------------------------
        # Create one-hot encoding for labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # If label is true class, use phi (margin applied), else use cosine
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Scale the logits by s
        output *= self.s

        return output


class EfficientNetArcFace(nn.Module):
    """
    Hotel Identification Model.
    Architecture: EfficientNet-B1 -> GeM Pooling -> Flatten -> Linear -> BN -> ArcFace
    """

    def __init__(
        self,
        n_classes=Config.NUM_CLASSES,
        model_name=Config.MODEL_NAME,
        embedding_size=Config.EMBEDDING_SIZE,
        pretrained=Config.PRETRAINED,
        s=Config.ARCFACE_SCALE,
        m=Config.ARCFACE_MARGIN,
        ls_eps=Config.ARCFACE_LS_EPS,
    ):
        super(EfficientNetArcFace, self).__init__()

        # 1. Backbone
        # num_classes=0 and global_pool='' ensures we get the raw spatial feature map
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Determine the output feature dimension of the backbone dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 256, 256)
            features = self.backbone(dummy_input)
            in_features = features.shape[1]

        # 2. Pooling Layer
        self.pooling = GeM()

        # 3. BN-Neck
        # Projects flattened features to embedding dimension and applies Batch Norm
        self.neck = nn.Sequential(
            nn.Linear(in_features, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size),
        )

        # 4. Classification Head (ArcFace)
        self.head = ArcMarginProduct(embedding_size, n_classes, s=s, m=m, ls_eps=ls_eps)

    def forward(self, x, labels=None):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W).
            labels (torch.Tensor, optional): Ground truth labels. Defaults to None.

        Returns:
            torch.Tensor:
                - If labels are provided (Training): Scaled logits from ArcFace head.
                - If labels are None (Inference): Embeddings from the BN-Neck.
        """
        # Feature Extraction
        features = self.backbone(x)

        # Pooling (B, C, H, W) -> (B, C, 1, 1)
        features = self.pooling(features)

        # Flatten (B, C, 1, 1) -> (B, C)
        features = features.flatten(1)

        # Neck (B, C) -> (B, Embedding_Size)
        embeddings = self.neck(features)

        if labels is not None:
            # Training Phase: Calculate ArcFace logits
            return self.head(embeddings, labels)
        else:
            # Inference Phase: Return embeddings for TTA/Ranking
            return embeddings
