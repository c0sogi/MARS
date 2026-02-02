import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
from library.config import Config


class ArcMarginProduct(nn.Module):
    r"""
    Implement of large margin arc distance:
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
        self.ls_eps = ls_eps  # label smoothing epsilon
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # input: (batch, embedding_size)
        # weight: (num_classes, embedding_size)
        # cosine: (batch, num_classes)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # If inference (no label), return scaled cosine similarity
        if label is None:
            return cosine * self.s

        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # --------------------------- Calculate output ---------------------------
        # output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # Optimized implementation:
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class WhaleModel(nn.Module):
    """
    Main model class for Whale Identification.
    Integrates a backbone, a projection neck, and an ArcFace head.
    """

    def __init__(self, model_name, num_classes, embedding_size=512, pretrained=True):
        super(WhaleModel, self).__init__()

        # Load Backbone using timm
        # num_classes=0 removes the classification head
        # global_pool='avg' ensures we get a pooled feature vector (B, num_features)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        in_features = self.backbone.num_features

        # Neck: Linear -> BN (No Dropout as per strategy to prevent underfitting)
        self.neck = nn.Sequential(
            nn.Linear(in_features, embedding_size), nn.BatchNorm1d(embedding_size)
        )

        # Head: ArcFace
        self.head = ArcMarginProduct(
            in_features=embedding_size,
            out_features=num_classes,
            s=Config.ARCFACE_S,
            m=Config.ARCFACE_M,
        )

    def forward(self, images, labels=None):
        """
        Forward pass.
        Args:
            images (torch.Tensor): Input images.
            labels (torch.Tensor, optional): Ground truth labels.
                                             If provided, returns ArcFace logits with margin.
                                             If None, returns scaled cosine similarities.
        """
        # Extract features
        features = self.backbone(images)

        # Neck
        embeddings = self.neck(features)

        # Head (ArcFace)
        logits = self.head(embeddings, labels)

        return logits


def get_model(model_name, num_classes=None, pretrained=True):
    """
    Factory function to create a WhaleModel instance.

    Args:
        model_name (str): Name of the backbone (e.g., 'densenet121', 'resnet50_ibn_a').
        num_classes (int, optional): Number of output classes. Defaults to Config.NUM_CLASSES.
        pretrained (bool): Whether to load pretrained backbone weights.

    Returns:
        WhaleModel: Instantiated model.
    """
    if num_classes is None:
        num_classes = Config.NUM_CLASSES

    model = WhaleModel(
        model_name=model_name,
        num_classes=num_classes,
        embedding_size=Config.EMBEDDING_SIZE,
        pretrained=pretrained,
    )

    return model
