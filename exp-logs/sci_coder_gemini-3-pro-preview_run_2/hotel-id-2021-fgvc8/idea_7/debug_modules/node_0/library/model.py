import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x: (B, C, H, W)
        # Output: (B, C, 1, 1)
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

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


class SubCenterArcMarginProduct(nn.Module):
    """
    ArcFace head with K sub-centers per class.
    """

    def __init__(
        self,
        in_features,
        out_features,
        k=3,
        s=30.0,
        m=0.50,
        easy_margin=False,
        ls_eps=0.0,
    ):
        super(SubCenterArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.s = s
        self.m = m
        self.ls_eps = ls_eps

        # Weight shape: (Class * K, Embedding_Dim)
        self.weight = nn.Parameter(torch.FloatTensor(out_features * k, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # input: (B, Embedding_Dim)
        # label: (B,)

        # --------------------------- cosine ---------------------------
        # Normalize features and weights
        # cosine shape: (B, Class * K)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # Reshape to (B, Class, K) and take max over sub-centers
        cosine = cosine.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine, dim=2)  # (B, Class)

        # --------------------------- arcface margin ---------------------------
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # --------------------------- label smoothing ---------------------------
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        # Apply margin only to the ground truth class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class HotelRecognitionModel(nn.Module):
    """
    Main model class integrating ConvNeXt backbone, GeM pooling, and Sub-Center ArcFace head.
    """

    def __init__(self):
        super(HotelRecognitionModel, self).__init__()

        # Backbone: ConvNeXt Small
        # num_classes=0 and global_pool="" ensures we get the feature map (B, C, H, W)
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="",
        )

        # Determine input features size dynamically
        if hasattr(self.backbone, "num_features"):
            in_features = self.backbone.num_features
        else:
            # Fallback
            in_features = 768  # Default for convnext_small

        # Pooling and Neck
        self.pooling = GeM()
        self.bn1 = nn.BatchNorm1d(in_features)
        self.dropout = nn.Dropout(Config.DROPOUT)
        self.fc = nn.Linear(in_features, Config.EMBEDDING_SIZE)
        self.bn2 = nn.BatchNorm1d(Config.EMBEDDING_SIZE)

        # Head (Metric Learning)
        self.head = SubCenterArcMarginProduct(
            in_features=Config.EMBEDDING_SIZE,
            out_features=Config.N_CLASSES,
            k=Config.K_SUB_CENTERS,
            s=Config.SCALE,
            m=Config.MARGIN,
            ls_eps=Config.LABEL_SMOOTHING,
        )

    def forward(self, x, labels=None):
        # Feature Extraction
        x = self.backbone(x)  # (B, C, H, W)

        # Pooling
        x = self.pooling(x).flatten(1)  # (B, C)

        # Neck
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn2(x)  # Embeddings (B, Embedding_Dim)

        if labels is not None:
            # Training: Return logits with margin
            return self.head(x, labels)

        # Inference: Return embeddings
        return x
