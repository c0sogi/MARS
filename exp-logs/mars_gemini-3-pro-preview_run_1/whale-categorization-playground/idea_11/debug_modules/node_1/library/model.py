import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class ElasticArcFace(nn.Module):
    """
    ElasticFace: Elastic Angular Margin Loss.

    This module implements the ElasticFace loss head, which replaces the fixed margin
    of ArcFace with a stochastic margin sampled from a Gaussian distribution N(m, sigma).
    This flexibility allows the model to better accommodate class overlap and singleton
    classes by relaxing the strict geometric constraints during training.

    Reference: "ElasticFace: Elastic Margin Loss for Deep Face Recognition"
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, std=0.05):
        """
        Args:
            in_features (int): Dimension of input embeddings.
            out_features (int): Number of classes.
            s (float): Scale factor.
            m (float): Mean margin.
            std (float): Standard deviation for the random margin.
        """
        super(ElasticArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.std = std

        # Weight matrix (Class centers)
        # Shape: (out_features, in_features) to match F.linear behavior
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels=None):
        """
        Args:
            embeddings (torch.Tensor): Input features of shape (Batch, in_features).
            labels (torch.Tensor, optional): Ground truth labels of shape (Batch,).

        Returns:
            torch.Tensor: Logits (scaled cosine similarities) of shape (Batch, out_features).
        """
        # 1. Normalize weights and input embeddings (L2 Norm)
        # F.normalize defaults to p=2, dim=1
        W = F.normalize(self.weight, p=2, dim=1)
        X = F.normalize(embeddings, p=2, dim=1)

        # 2. Compute Cosine Similarity
        # Result: (Batch, Out_Features)
        cosine = F.linear(X, W)

        # 3. Inference / Validation Mode
        # If no labels are provided or we are not training, return scaled cosine logits.
        if labels is None or not self.training:
            return cosine * self.s

        # 4. Training Mode: Apply Elastic Margin

        # Clamp cosine for numerical stability in acos
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Sample random margin from N(0, 1) * std + m
        # We sample a single random margin for the batch for efficiency,
        # or we could sample per instance. Per-batch is standard for efficiency.
        eps = torch.randn(1, device=cosine.device) * self.std
        m_random = self.m + eps

        # Get the cosine value corresponding to the ground truth class
        # labels: (Batch,) -> index: (Batch, 1)
        index = labels.view(-1, 1).long()
        cosine_target = cosine.gather(1, index)

        # Compute theta = arccos(cos_theta)
        theta = cosine_target.acos()

        # Apply elastic margin: cos(theta + m_random)
        cosine_target_margin = torch.cos(theta + m_random)

        # Create one-hot mask to replace only the target class logits
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, index, 1.0)

        # Combine: Use modified cosine for target class, original cosine for others
        logits = cosine * (1 - one_hot) + cosine_target_margin * one_hot

        # Scale logits
        logits = logits * self.s

        return logits


class WhaleDenseNet(nn.Module):
    """
    Ensemble Component Model: DenseNet121 with ElasticArcFace.

    Architecture:
    1. Backbone: DenseNet121 (pre-trained).
    2. Pooling: Global Average Pooling (GAP).
    3. Neck: BN -> Linear(512) -> BN (No Dropout).
    4. Head: ElasticArcFace.
    """

    def __init__(
        self,
        num_classes=Config.NUM_CLASSES,
        embedding_dim=Config.EMBEDDING_DIM,
        pretrained=True,
    ):
        super(WhaleDenseNet, self).__init__()

        # 1. Backbone
        weights = "DEFAULT" if pretrained else None
        self.backbone = models.densenet121(weights=weights)

        # Capture input features of the original classifier (1024 for DenseNet121)
        in_features = self.backbone.classifier.in_features

        # Remove original classifier to save memory/parameters
        del self.backbone.classifier

        # 2. Neck
        # Learnable bottleneck to adapt semantic features to geometric metric space
        # We explicitly exclude Dropout as it conflicts with margin-based losses
        self.neck = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        # 3. Head
        self.head = ElasticArcFace(
            in_features=embedding_dim,
            out_features=num_classes,
            s=Config.S,
            m=Config.M,
            std=Config.SIGMA,
        )

    def forward(self, x, labels=None):
        """
        Args:
            x (torch.Tensor): Input images (Batch, 3, H, W).
            labels (torch.Tensor, optional): Target labels.
        """
        # Extract features
        # DenseNet features output: (B, 1024, H/32, W/32)
        # Note: torchvision's densenet features() ends with a BatchNorm (norm5).
        features = self.backbone.features(x)

        # Apply ReLU and Global Average Pooling
        # (Standard DenseNet termination before classifier)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)  # (B, 1024)

        # Apply Neck to get embeddings
        embeddings = self.neck(out)  # (B, 512)

        # Apply Head to get logits
        logits = self.head(embeddings, labels)

        return logits

    def get_embedding(self, x):
        """
        Utility method to extract embeddings without passing through the head.
        Useful for debugging or KNN-based inference strategies.
        """
        features = self.backbone.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        embeddings = self.neck(out)
        return embeddings
