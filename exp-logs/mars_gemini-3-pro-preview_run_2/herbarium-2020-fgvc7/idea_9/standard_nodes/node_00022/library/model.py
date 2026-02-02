import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Interpolates between Average Pooling (p=1) and Max Pooling (p=infinity).
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        # Clamp to avoid NaN in power operation
        x = x.clamp(min=self.eps)
        # Apply GeM formula: (AvgPool(x^p))^(1/p)
        return F.avg_pool2d(x.pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p)


class ArcFaceHead(nn.Module):
    """
    ArcFace (Additive Angular Margin) Classification Head.
    """

    def __init__(self, in_features, out_features, scale=30.0, margin=0.50):
        super(ArcFaceHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin

        # Weight matrix for the classification layer (Centers)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precompute constants for the margin logic
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        # Threshold for numerical stability
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label=None):
        # input: (Batch, Embedding_Size)
        # label: (Batch,) - Ground truth class indices

        # 1. Normalize Features and Weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # 2. If Training (label provided), apply margin
        if label is not None:
            # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
            phi = cosine * self.cos_m - sine * self.sin_m

            # Numerical stability: keep phi only where cosine > th
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

            # Create one-hot encoding to apply margin only to the target class
            one_hot = torch.zeros(cosine.size(), device=input.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)

            # output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

            # Scale the logits
            output *= self.scale
        else:
            # Inference: just scale the cosine similarities
            output = cosine * self.scale

        return output


class HerbariumNet(nn.Module):
    """
    Main Model Architecture: EfficientNet-B3 + GeM + ArcFace
    """

    def __init__(self, pretrained=True):
        super(HerbariumNet, self).__init__()

        # 1. Backbone: EfficientNet-B3
        # num_classes=0 and global_pool='' ensures we get the spatial feature map
        self.backbone = timm.create_model(
            Config.BACKBONE, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Determine input channels dynamically
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.backbone(dummy_input)
            in_channels = features.shape[1]

        # 2. Pooling: Generalized Mean Pooling
        self.pooling = GeM(p=Config.GEM_P)

        # 3. Neck: Projection to Embedding Space
        # BatchNorm -> Dropout -> Linear -> BatchNorm helps stabilize ArcFace training
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.dropout = nn.Dropout(p=Config.DROPOUT_RATE)
        self.fc = nn.Linear(in_channels, Config.EMBEDDING_SIZE)
        self.bn2 = nn.BatchNorm1d(Config.EMBEDDING_SIZE)

        # 4. Head: ArcFace
        self.head = ArcFaceHead(
            in_features=Config.EMBEDDING_SIZE,
            out_features=Config.NUM_CLASSES,
            scale=Config.ARCFACE_SCALE,
            margin=Config.ARCFACE_MARGIN,
        )

    def forward(self, x, label=None):
        # Feature Extraction
        x = self.backbone(x)  # (B, C, H, W)

        # Pooling
        x = self.pooling(x)  # (B, C, 1, 1)
        x = x.flatten(1)  # (B, C)

        # Neck / Projection
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn2(x)  # (B, Embedding_Size)

        # Classification Head
        # Returns scaled logits (cosine similarity)
        x = self.head(x, label)

        return x
