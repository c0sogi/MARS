import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models
from library.config import Config


class ArcMarginProduct(nn.Module):
    r"""
    Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)
    """

    def __init__(
        self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = torch.zeros(cosine.size(), device=input.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            return self.s * (one_hot * phi + (1.0 - one_hot) * cosine)
        else:
            one_hot = torch.zeros(cosine.size(), device=input.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.s
            return output


class WhaleDenseNet(nn.Module):
    """
    DenseNet121 backbone with a projection head and ArcFace loss.
    """

    def __init__(
        self,
        num_classes=Config.NUM_CLASSES,
        embedding_size=Config.EMBEDDING_SIZE,
        pretrained=Config.PRETRAINED,
        dropout_rate=Config.DROPOUT_RATE,
        s=Config.ARCFACE_SCALE,
        m=Config.ARCFACE_MARGIN,
    ):
        super(WhaleDenseNet, self).__init__()

        # Load Backbone
        weights = "DEFAULT" if pretrained else None
        self.backbone = models.densenet121(weights=weights)

        # Get feature dimension before the original classifier
        in_features = self.backbone.classifier.in_features

        # We replace the classifier logic in forward, but keeping the attribute clean is good practice
        self.backbone.classifier = nn.Identity()

        # Projection Neck
        # BN -> Dropout -> Linear -> BN
        # This helps adapt features to the hypersphere constraints of ArcFace
        self.neck = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

        # ArcFace Head
        self.arcface = ArcMarginProduct(
            in_features=embedding_size, out_features=num_classes, s=s, m=m
        )

    def forward(self, x, labels=None):
        # Extract features from backbone
        # DenseNet features: (B, C, H, W)
        features = self.backbone.features(x)

        # Standard DenseNet pooling logic: ReLU -> GAP -> Flatten
        features = F.relu(features, inplace=True)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)

        # Pass through projection neck
        embeddings = self.neck(features)

        if labels is not None:
            # Training: Return ArcFace logits
            return self.arcface(embeddings, labels)
        else:
            # Inference: Return embeddings
            return embeddings

    def get_embedding(self, x):
        """Helper for inference to explicitly get embeddings."""
        return self.forward(x, labels=None)
