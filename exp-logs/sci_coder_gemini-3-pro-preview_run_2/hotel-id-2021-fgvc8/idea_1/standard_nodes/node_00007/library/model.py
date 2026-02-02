import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models import resnet18, ResNet18_Weights
from library.config import Config


class ArcMarginProduct(nn.Module):
    """
    ArcFace (Additive Angular Margin Loss) layer.
    Cite solution_lesson_node_00001: Pivot to Metric Learning for high-cardinality.
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # If inference (no label), return scaled cosine similarity
        if label is None:
            return cosine * self.s

        # --------------------------- Training with Margin ---------------------------
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class HotelClassifier(nn.Module):
    """
    Hotel ID Classifier using a ResNet-18 backbone with ArcFace head.
    """

    def __init__(
        self,
        n_classes: int = Config.NUM_CLASSES,
        pretrained: bool = Config.PRETRAINED,
        dropout: float = Config.DROPOUT,
    ):
        super(HotelClassifier, self).__init__()

        # Select weights based on pretrained flag
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        # Load the ResNet-18 backbone
        self.backbone = resnet18(weights=weights)
        in_features = self.backbone.fc.in_features

        # Remove original fc
        self.backbone.fc = nn.Identity()

        # Embedding layer
        self.bn1 = nn.BatchNorm1d(in_features)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_features, Config.EMBEDDING_SIZE)
        self.bn2 = nn.BatchNorm1d(Config.EMBEDDING_SIZE)

        # ArcFace Head
        self.arcface = ArcMarginProduct(
            in_features=Config.EMBEDDING_SIZE,
            out_features=n_classes,
            s=Config.ARCFACE_S,
            m=Config.ARCFACE_M,
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input images
            labels: Ground truth labels (optional, for training)
        """
        x = self.backbone(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn2(x)

        return self.arcface(x, labels)
