import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes (1/|X| * sum(x^p))^(1/p).
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x: (B, C, H, W)
        # Clamp to avoid NaN gradients with power
        x = x.clamp(min=eps)
        # Average pooling calculates sum(x^p) / (H*W)
        # Then take the (1/p)-th power
        return F.avg_pool2d(x.pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class ArcMarginProduct(nn.Module):
    """
    ArcFace (Additive Angular Margin) classification head.
    Applies margin penalty to the angle between embeddings and class centers.
    """

    def __init__(
        self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps

        # Learnable class centers (Weights)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # input: (B, in_features) - L2 normalized embeddings
        # label: (B,) - ground truth class indices

        # 1. Normalize weights
        W = F.normalize(self.weight)

        # 2. Normalize input (ensure input is on hypersphere)
        X = F.normalize(input)

        # 3. Compute Cosine Similarity
        cosine = F.linear(X, W)

        # 4. Apply Angular Margin
        # sin(theta) = sqrt(1 - cos(theta)^2)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Handle condition where theta + m > pi
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 5. Apply margin only to ground truth classes
        # Create one-hot mask
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # output = one_hot * phi + (1 - one_hot) * cosine
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # 6. Scale logits
        output *= self.s

        return output


class HotelRecognitionModel(nn.Module):
    """
    Main model class for Hotel Identification.
    Architecture: Backbone (ResNet50) -> GeM -> Neck -> L2 Norm -> [ArcFace Head].
    """

    def __init__(
        self,
        n_classes=Config.num_classes,
        backbone_name=Config.backbone_name,
        pretrained=Config.pretrained,
        embedding_size=Config.embedding_size,
    ):
        super(HotelRecognitionModel, self).__init__()

        # Backbone: ResNet50 (or other) without classification head
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Dynamically determine backbone output channels
        with torch.no_grad():
            dummy = torch.zeros(1, 3, Config.crop_size, Config.crop_size)
            feats = self.backbone(dummy)
            in_features = feats.shape[1]

        # Pooling Layer
        self.gem = GeM()

        # Neck (Projection Head)
        self.neck = nn.Sequential(
            nn.Linear(in_features, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.PReLU(),
        )

        # ArcFace Head (Used only during training)
        self.arcface = ArcMarginProduct(
            in_features=embedding_size,
            out_features=n_classes,
            s=Config.s,
            m=Config.m,
            easy_margin=Config.easy_margin,
            ls_eps=Config.ls_eps,
        )

    def forward(self, x, labels=None):
        """
        Args:
            x (torch.Tensor): Input images (B, 3, H, W)
            labels (torch.Tensor, optional): Ground truth labels (B,).
                                             If provided, returns ArcFace logits for loss calculation.
                                             If None, returns L2-normalized embeddings for inference.
        """
        # 1. Feature Extraction
        x = self.backbone(x)  # (B, C, H, W)

        # 2. Pooling
        x = self.gem(x)  # (B, C, 1, 1)
        x = x.view(x.size(0), -1)  # (B, C)

        # 3. Projection
        x = self.neck(x)  # (B, embedding_size)

        # 4. L2 Normalization
        # Essential for Cosine Similarity based Metric Learning
        embedding = F.normalize(x, p=2, dim=1)

        # 5. Conditional Output
        if labels is not None:
            # Training: Pass through ArcFace to get logits
            logits = self.arcface(embedding, labels)
            return logits

        # Inference: Return embeddings
        return embedding
