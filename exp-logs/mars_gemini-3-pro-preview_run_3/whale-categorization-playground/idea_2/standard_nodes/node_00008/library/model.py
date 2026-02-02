import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import library.config as config


class ArcMarginProduct(nn.Module):
    r"""
    Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        easy_margin: optimization strategy
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

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # Normalize weights and input features
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # Calculate sin(theta)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

        # Calculate cos(theta + m) using formula: cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Keep phi only when cosine > th, else penalize with mm
            # This handles the condition where theta + m > pi
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # --------------------------- torch.where(out_i = {x_i if condition_i else y_i}) ---------------------------
        # Add margin only to the ground truth class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class WhaleDenseNet(nn.Module):
    """
    DenseNet121 Backbone with ArcFace Head for Whale Identification.
    """

    def __init__(
        self, num_classes, embedding_size=config.EMBEDDING_SIZE, pretrained=True
    ):
        super(WhaleDenseNet, self).__init__()

        # Load Pretrained DenseNet121
        # Using weights='DEFAULT' if available in newer torchvision, else pretrained=True
        try:
            # For newer torchvision versions
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            self.backbone = models.densenet121(weights=weights)
        except AttributeError:
            # Fallback for older torchvision versions
            self.backbone = models.densenet121(pretrained=pretrained)

        # Extract features (conv layers)
        self.features = self.backbone.features

        # DenseNet121 outputs 1024 channels at the last conv layer
        self.num_features = 1024

        # Neck: BN -> Dropout -> FC -> BN
        # We use a BN layer immediately after features to normalize statistics
        self.bn1 = nn.BatchNorm1d(self.num_features)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.num_features, embedding_size)
        self.bn2 = nn.BatchNorm1d(embedding_size)

        # Head: ArcFace
        self.arcface = ArcMarginProduct(
            in_features=embedding_size,
            out_features=num_classes,
            s=config.ARC_SCALE,
            m=config.ARC_MARGIN,
        )

    def forward(self, x, labels=None):
        """
        Args:
            x (torch.Tensor): Input images [B, C, H, W]
            labels (torch.Tensor, optional): Ground truth labels [B].
                                             If provided, returns ArcFace logits (Training).
                                             If None, returns normalized embeddings (Inference).
        """
        # Feature Extraction
        x = self.features(x)
        x = F.relu(x, inplace=True)

        # Global Average Pooling
        # x shape: [B, 1024, H', W'] -> [B, 1024, 1, 1]
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        # Neck
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn2(x)

        # If labels are provided, we are in training mode and need ArcFace logits
        if labels is not None:
            logits = self.arcface(x, labels)
            return logits

        # If labels are None, we are in inference mode and need normalized embeddings
        # ArcFace relies on Cosine Similarity, so embeddings must be L2 normalized
        return F.normalize(x)
