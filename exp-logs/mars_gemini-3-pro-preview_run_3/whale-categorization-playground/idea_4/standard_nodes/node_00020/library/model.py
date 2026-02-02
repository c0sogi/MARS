import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
from library.config import Config


def l2_norm(input, axis=1):
    """
    Computes L2 normalization of input tensor.
    Returns normalized tensor and the norms.
    """
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm + 1e-6)
    return output, norm


class AdaFaceHead(nn.Module):
    """
    AdaFace: Quality Adaptive Margin for Face Recognition.
    Adjusts margin based on image quality (feature norm).
    """

    def __init__(
        self, embedding_size, num_classes, m=0.5, h=0.333, s=64.0, t_alpha=0.01
    ):
        super(AdaFaceHead, self).__init__()
        self.class_num = num_classes
        self.kernel = nn.Parameter(torch.Tensor(embedding_size, num_classes))

        # Initialize kernel
        nn.init.xavier_uniform_(self.kernel)

        self.m = m
        self.h = h
        self.s = s
        self.t_alpha = t_alpha
        self.eps = 1e-3

        # Register buffers for batch statistics (non-trainable, persistent)
        self.register_buffer("batch_mean", torch.tensor(20.0))
        self.register_buffer("batch_std", torch.tensor(100.0))

    def forward(self, features, targets):
        # 1. Normalize weights
        kernel_norm = F.normalize(self.kernel, p=2, dim=0)

        # 2. Normalize features and get norms
        features_norm, norms = l2_norm(features, axis=1)

        # 3. Calculate Cosine Similarity
        cosine = torch.mm(features_norm, kernel_norm)
        # Clamp for numerical stability in acos
        cosine = cosine.clamp(-1 + self.eps, 1 - self.eps)

        # 4. Adaptive Margin Calculation
        # Detach norms to stop gradient backprop into the margin calculation itself
        safe_norms = norms.detach().clamp(min=0.001, max=100)

        if self.training:
            with torch.no_grad():
                mean = safe_norms.mean()
                std = safe_norms.std()
                self.batch_mean = (
                    1 - self.t_alpha
                ) * self.batch_mean + self.t_alpha * mean
                self.batch_std = (
                    1 - self.t_alpha
                ) * self.batch_std + self.t_alpha * std

        # Standardize norms: g(z)
        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std + self.eps)
        margin_scaler = margin_scaler * self.h
        margin_scaler = torch.clamp(margin_scaler, -1.0, 1.0)

        # m_adapt = m * g(z)
        m_adapt = self.m * margin_scaler

        # 5. Apply Margin to Target Class
        # Get theta
        theta = torch.acos(cosine)

        # Create margin modifier matrix
        margin_add = torch.zeros_like(cosine)
        # Scatter m_adapt to the specific target indices
        margin_add.scatter_(1, targets.view(-1, 1), m_adapt)

        # Apply margin: theta + m_adapt
        # Note: If image is bad (m_adapt < 0), margin is subtracted (easier).
        # If image is good (m_adapt > 0), margin is added (harder).
        theta_m = theta + margin_add
        theta_m = theta_m.clamp(min=self.eps, max=math.pi - self.eps)

        # Convert back to cosine
        logits = torch.cos(theta_m) * self.s

        return logits


class WhaleModel(nn.Module):
    """
    EfficientNet-B2 backbone with AdaFace Head.
    """

    def __init__(self, num_classes=None):
        super(WhaleModel, self).__init__()

        # Backbone
        # num_classes=0 removes the classifier
        # global_pool='' removes the pooling layer, giving us flexibility
        self.backbone = timm.create_model(
            Config.model_name, pretrained=True, num_classes=0, global_pool=""
        )

        # Determine input features dynamically
        # EfficientNet-B2 is 1408, but this makes it robust to config changes
        with torch.no_grad():
            dummy = torch.randn(1, 3, Config.input_size, Config.input_size)
            features = self.backbone(dummy)
            self.in_features = features.shape[1]

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Neck / Embedding Layer
        # BN -> Dropout -> Linear -> BN (Common for ArcFace/AdaFace)
        self.neck = nn.Sequential(
            nn.BatchNorm1d(self.in_features),
            nn.Dropout(Config.drop_rate),
            nn.Linear(self.in_features, Config.embedding_dim),
            nn.BatchNorm1d(Config.embedding_dim),
        )

        # Head (only needed if num_classes is provided)
        if num_classes is not None:
            self.head = AdaFaceHead(
                embedding_size=Config.embedding_dim,
                num_classes=num_classes,
                m=Config.adaface_margin,
                h=Config.adaface_h,
                s=Config.adaface_scale,
            )
        else:
            self.head = None

    def forward(self, x, targets=None):
        # 1. Backbone Feature Extraction
        x = self.backbone(x)

        # 2. Pooling
        x = self.global_pool(x)
        x = x.flatten(1)

        # 3. Projection Neck
        embeddings = self.neck(x)

        # 4. Head / Output
        if self.training and targets is not None:
            if self.head is None:
                raise ValueError(
                    "Model was initialized without num_classes, but called in training mode with targets."
                )
            logits = self.head(embeddings, targets)
            return logits
        else:
            # Inference: Return normalized embeddings
            return F.normalize(embeddings, p=2, dim=1)
