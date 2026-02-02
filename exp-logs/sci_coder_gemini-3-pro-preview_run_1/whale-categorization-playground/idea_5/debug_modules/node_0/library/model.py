import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models
from library.config import Config


class ArcMarginProduct(nn.Module):
    """
    Implementation of ArcFace (Additive Angular Margin Loss).
    Reference: https://arxiv.org/abs/1801.07698
    """

    def __init__(
        self,
        in_features,
        out_features,
        s=30.0,
        m=0.50,
        easy_margin=False,
        ls_eps=0.0,
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps
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

        # If we are in inference mode (label is None), return the raw cosine similarities
        # or scaled logits. Ranking is preserved either way.
        if label is None:
            return cosine * self.s

        # --------------------------- convert label to one-hot ---------------------------
        # Calculate sin(theta)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # Calculate phi = cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Handle numerical stability for theta > pi - m
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        # Create one_hot encoding
        # one_hot = torch.zeros(cosine.size(), device=Config.DEVICE)
        # one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Efficient implementation without creating full one-hot matrix
        # Select the cosine values corresponding to the ground truth labels
        # and replace them with phi (the penalized cosine)

        # Create a copy to modify
        output = cosine.clone()

        # Get indices
        indices = torch.arange(0, len(label), device=input.device, dtype=torch.long)

        # Update the specific indices with the margin penalty
        output[indices, label] = phi

        # Scale the result
        output *= self.s

        return output


class WhaleArcFaceModel(nn.Module):
    """
    DenseNet169 Backbone with a Projection Head and ArcFace Loss.
    """

    def __init__(
        self,
        num_classes=Config.NUM_CLASSES,
        backbone_name=Config.BACKBONE,
        embedding_dim=Config.EMBEDDING_DIM,
        dropout_rate=Config.DROPOUT_RATE,
        s=Config.ARCFACE_SCALE,
        m=Config.ARCFACE_MARGIN,
        pretrained=True,
    ):
        super(WhaleArcFaceModel, self).__init__()

        # 1. Backbone: DenseNet169
        # We use the features section of the densenet
        if backbone_name == "densenet169":
            weights = "DEFAULT" if pretrained else None
            self.backbone = models.densenet169(weights=weights)
            in_features = self.backbone.classifier.in_features  # 1664 for DenseNet169
            self.features = self.backbone.features
        else:
            # Fallback or extension point for other backbones
            weights = "DEFAULT" if pretrained else None
            self.backbone = models.densenet121(weights=weights)
            in_features = self.backbone.classifier.in_features
            self.features = self.backbone.features

        # 2. Pooling
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        # 3. Neck (Projection Head)
        # Structure: BN -> Dropout -> Linear(Features -> 512) -> BN
        self.neck = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )

        # 4. Head (ArcFace)
        self.arcface = ArcMarginProduct(
            in_features=embedding_dim, out_features=num_classes, s=s, m=m
        )

    def forward(self, images, labels=None):
        """
        Args:
            images (torch.Tensor): Input images (B, C, H, W)
            labels (torch.Tensor, optional): Ground truth labels for training.
                                             If None, returns inference logits.

        Returns:
            torch.Tensor: Logits (B, Num_Classes)
        """
        # Extract features from backbone
        # Shape: (B, 1664, H_feat, W_feat)
        x = self.features(images)

        # Global Average Pooling
        # Shape: (B, 1664, 1, 1)
        x = self.pooling(x)

        # Flatten
        # Shape: (B, 1664)
        x = torch.flatten(x, 1)

        # Projection Neck
        # Shape: (B, 512)
        embeddings = self.neck(x)

        # ArcFace Head
        # If labels are provided (Training), applies margin penalty
        # If labels are None (Inference), returns scaled cosine similarities
        logits = self.arcface(embeddings, labels)

        return logits
