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
        # Clamp for numerical stability
        x = x.clamp(min=eps).pow(p)
        # Average pooling over spatial dimensions
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        # Root p
        x = x.pow(1.0 / p)
        # Flatten to (B, C)
        return x.flatten(1)

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

        # Weights for class centers
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # 1. Normalize inputs and weights
        # input: (B, in_features)
        # weight: (out_features, in_features)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # If no label is provided (inference), return scaled cosine similarity
        if label is None:
            return cosine * self.s

        # 2. Apply ArcFace margin logic
        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 3. Create one-hot encoding to apply margin only to ground truth classes
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Optional: Label Smoothing integration
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        # 4. Combine: use phi for target class, cosine for others
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # 5. Scale
        output *= self.s

        return output


class HotelIdModel(nn.Module):
    def __init__(self):
        super(HotelIdModel, self).__init__()

        # 1. Backbone: ConvNeXt-Tiny
        # num_classes=0 and global_pool='' ensures we get spatial feature maps (B, C, H, W)
        self.backbone = timm.create_model(
            Config.backbone, pretrained=Config.pretrained, num_classes=0, global_pool=""
        )

        # Determine backbone output features dynamically
        with torch.no_grad():
            dummy = torch.randn(1, 3, Config.image_size, Config.image_size)
            features = self.backbone(dummy)
            in_features = features.shape[1]

        # 2. Pooling: GeM
        self.pooling = GeM()

        # 3. Neck: Linear Projection + BN
        self.fc = nn.Linear(in_features, Config.embedding_size)
        self.bn = nn.BatchNorm1d(Config.embedding_size)

        # 4. Head: ArcFace
        self.arcface = ArcMarginProduct(
            in_features=Config.embedding_size,
            out_features=Config.n_classes,
            s=Config.arcface_s,
            m=Config.arcface_m,
            ls_eps=Config.arcface_ls_eps,
        )

        self._init_params()

    def _init_params(self):
        # Initialize Neck layers
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def extract_features(self, x):
        """
        Extracts embeddings for Test-Time Augmentation (TTA).
        Returns the feature vector after the BN-Neck.
        """
        x = self.backbone(x)
        x = self.pooling(x)
        x = self.fc(x)
        x = self.bn(x)
        return x

    def forward(self, x, labels=None):
        """
        Forward pass.
        Args:
            x: Input images
            labels: Ground truth labels (optional)
        Returns:
            logits: ArcFace logits (if labels provided) or Cosine similarities (if None)
        """
        features = self.extract_features(x)
        logits = self.arcface(features, labels)
        return logits
