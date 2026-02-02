import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from library.config import Config


class ArcMarginProduct(nn.Module):
    """
    ArcFace output layer (Cite solution_lesson_node_00001).
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label=None):
        # Normalize features and weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # If inference (no label), return scaled cosine similarity
        if label is None:
            return cosine * self.s

        # Training: Add margin
        # clamp for numerical stability
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cosine)
        target_logits = torch.cos(theta + self.m)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = cosine * (1 - one_hot) + target_logits * one_hot
        output *= self.s
        return output


class HotelClassifier(nn.Module):
    """
    Hotel ID Classifier using a ResNet-50 backbone with ArcFace head.
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
            weights = ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None

        # Load the ResNet-50 backbone
        self.backbone = resnet50(weights=weights)

        # Get the input dimension of the original fully connected layer (2048 for ResNet50)
        in_features = self.backbone.fc.in_features

        # Remove original FC
        self.backbone.fc = nn.Identity()

        # Embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(in_features, 512), nn.BatchNorm1d(512), nn.Dropout(p=dropout)
        )

        # ArcFace Head
        self.arc = ArcMarginProduct(512, n_classes, s=Config.ARC_S, m=Config.ARC_M)

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x: Input images
            labels: Ground truth labels (optional, for training only)
        """
        x = self.backbone(x)
        x = self.embedding(x)
        return self.arc(x, labels)
