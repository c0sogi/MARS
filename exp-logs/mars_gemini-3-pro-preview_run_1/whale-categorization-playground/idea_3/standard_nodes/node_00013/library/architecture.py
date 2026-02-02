import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.configuration import Config


class ArcMarginProduct(nn.Module):
    """
    Implementation of ArcFace (Additive Angular Margin Loss).
    Reference: https://arxiv.org/abs/1801.07698
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
        # Normalize weights and input features
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # If no label is provided (Inference mode), return the scaled cosine similarities
        if label is None:
            return cosine * self.s

        # --------------------------- Training Mode ---------------------------
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

        # cos(theta + m) = cos(theta) * cos(m) - sin(theta) * sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Handle numerical stability for angles > pi - m
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- Convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # --------------------------- Calculate output ---------------------------
        # Add margin only to the ground truth class
        # output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class WhaleArcFaceModel(nn.Module):
    """
    DenseNet121 backbone with an ArcFace head for fine-grained whale identification.
    """

    def __init__(self, num_classes=None):
        super(WhaleArcFaceModel, self).__init__()

        # Use Config.num_classes if not explicitly provided
        if num_classes is None:
            num_classes = Config.num_classes

        # 1. Backbone: DenseNet121
        # Using default weights (ImageNet)
        self.backbone = models.densenet121(weights="DEFAULT")

        # Extract features (DenseNet features are in .features)
        # DenseNet121 outputs 1024 channels at the last layer
        self.features = self.backbone.features
        in_features = 1024

        # 2. Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 3. Embedding Layer (Bottleneck)
        # Projects 1024 -> 512 (Config.embedding_size)
        self.dropout = nn.Dropout(p=Config.dropout_rate)
        self.embedding_layer = nn.Linear(in_features, Config.embedding_size)
        self.bn = nn.BatchNorm1d(Config.embedding_size)

        # 4. ArcFace Head
        self.arcface = ArcMarginProduct(
            in_features=Config.embedding_size,
            out_features=num_classes,
            s=Config.arcface_s,
            m=Config.arcface_m,
        )

    def forward(self, images, labels=None):
        """
        Args:
            images (torch.Tensor): Input images of shape (B, C, H, W)
            labels (torch.Tensor, optional): Target labels of shape (B,).
                                             If None, returns inference logits.
        """
        # Feature extraction
        x = self.features(images)
        x = self.gap(x)
        x = torch.flatten(x, 1)

        # Embedding projection
        x = self.dropout(x)
        x = self.embedding_layer(x)
        x = self.bn(x)

        # ArcFace classification
        # If labels are None, arcface returns scaled cosine similarities
        output = self.arcface(x, labels)

        return output
