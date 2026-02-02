import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models
from library.config import Config


class ArcMarginProduct(nn.Module):
    r"""
    Implement of large margin cosine distance:
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

        # Weight shape: (out_features, in_features)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # Normalize input and weights to get cosine similarity
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # Inference mode: return scaled cosine similarity
        if label is None:
            return cosine * self.s

        # Training mode: Apply angular margin
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Keep gradients stable
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        # Apply margin only to ground truth classes
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Scale the logits
        output *= self.s

        return output


class WhaleDenseNet(nn.Module):
    def __init__(self):
        super(WhaleDenseNet, self).__init__()

        # Hyperparameters
        self.num_classes = Config.NUM_CLASSES
        self.embedding_size = Config.EMBEDDING_SIZE
        self.backbone_name = Config.BACKBONE
        self.pretrained = Config.PRETRAINED
        self.dropout_rate = Config.PROJECTION_DROPOUT
        self.s = Config.ARCFACE_SCALE
        self.m = Config.ARCFACE_MARGIN
        self.ls_eps = Config.LABEL_SMOOTHING

        # 1. Backbone
        if self.backbone_name == "densenet169":
            weights = models.DenseNet169_Weights.DEFAULT if self.pretrained else None
            backbone = models.densenet169(weights=weights)
            in_features = (
                backbone.classifier.in_features
            )  # Typically 1664 for DenseNet169
            self.features = backbone.features
        else:
            raise ValueError(
                f"Backbone {self.backbone_name} is not supported in this implementation."
            )

        # 2. Neck (Projection Head)
        # Structure: BN -> Dropout -> Linear -> BN
        # This adapts features for the angular margin head
        self.neck = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(in_features, self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
        )

        # 3. Head (ArcFace)
        self.head = ArcMarginProduct(
            in_features=self.embedding_size,
            out_features=self.num_classes,
            s=self.s,
            m=self.m,
            easy_margin=False,
            ls_eps=self.ls_eps,
        )

    def forward(self, x, labels=None):
        # Extract features from backbone
        x = self.features(x)

        # DenseNet standard post-processing: ReLU -> Adaptive Avg Pool -> Flatten
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        # Apply Projection Neck
        embeddings = self.neck(x)

        # Apply ArcFace Head
        # If labels are provided, returns logits with margin (Training)
        # If labels are None, returns scaled cosine similarities (Inference)
        logits = self.head(embeddings, labels)

        return logits
