import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class ArcMarginProduct(nn.Module):
    """
    ArcFace (Additive Angular Margin Loss) Layer.
    Computes cos(theta + m) for the target class to enforce intra-class compactness.
    """

    def __init__(
        self,
        in_features,
        out_features,
        s=30.0,
        m=0.50,
        easy_margin=False,
        device=None,
        dtype=None,
    ):
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
        # Normalize input features and weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # If no label is provided (inference mode), return scaled cosine similarity
        if label is None:
            return cosine * self.s

        # --------------------------- Training Mode ---------------------------
        # sine = sqrt(1 - cos^2)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

        # phi = cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Handle numerical stability for angles > pi - m
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- Convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # --------------------------- Calculate output ---------------------------
        # Add margin only to the target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class WhaleDenseNet(nn.Module):
    """
    DenseNet121 backbone with a Projection Neck and ArcFace Head.
    Strategy:
    1. Backbone: DenseNet121 (Pretrained, Global Average Pooling)
    2. Neck: Linear -> BatchNorm (No Dropout)
    3. Head: ArcFace
    """

    def __init__(
        self,
        backbone_name=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
        embedding_size=Config.EMBEDDING_SIZE,
        num_classes=Config.NUM_CLASSES,
        s=Config.ARCFACE_SCALE,
        m=Config.ARCFACE_MARGIN,
    ):
        super(WhaleDenseNet, self).__init__()

        # 1. Backbone
        # num_classes=0 ensures we get the feature vector after pooling
        # global_pool='avg' ensures Global Average Pooling is applied
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # Determine input feature dimension dynamically
        # DenseNet121 typically has 1024 features
        in_features = self.backbone.num_features

        # 2. Projection Neck
        # Linear -> BatchNorm. Dropout is explicitly disabled per strategy.
        self.neck = nn.Sequential(
            nn.Linear(in_features, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size),
        )

        # 3. ArcFace Head
        self.head = ArcMarginProduct(
            in_features=embedding_size,
            out_features=num_classes,
            s=s,
            m=m,
        )

    def forward(self, x, labels=None):
        """
        Args:
            x (torch.Tensor): Input images (Batch, Channels, Height, Width)
            labels (torch.Tensor, optional): Target labels for ArcFace margin calculation.
                                             Required during training.
        Returns:
            logits (torch.Tensor): Scaled logits (Batch, Num_Classes)
        """
        # Extract features from backbone
        features = self.backbone(x)

        # Project features through the neck
        embeddings = self.neck(features)

        # Pass through ArcFace head
        # If labels are None, this returns scaled cosine similarity
        logits = self.head(embeddings, labels)

        return logits
