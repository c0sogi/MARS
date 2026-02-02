import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
from library.config import Config


class ArcMarginProduct(nn.Module):
    """
    Implement of large margin cosine distance:
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)
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
        # input: (batch, dim), weight: (out_dim, dim)
        # cosine: (batch, out_dim)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # If no label is provided, we are in inference mode.
        # Return scaled cosine similarities as logits.
        if label is None:
            return cosine * self.s

        # --------------------------- Training Logic ---------------------------
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Check if theta > pi - m to ensure monotonicity / stability
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- Convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # --------------------------- Calculate output ---------------------------
        # ArcFace: output = s * (cos(theta + m) for target class, cos(theta) for others)
        # (1 - one_hot) * cosine  -> Keeps original cosine for non-target classes
        # one_hot * phi           -> Applies margin penalty for target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class WhaleDenseNet(nn.Module):
    """
    DenseNet121 Backbone with a Projection Neck and ArcFace Head.
    """

    def __init__(self, model_name=None, pretrained=True):
        super(WhaleDenseNet, self).__init__()

        if model_name is None:
            model_name = Config.BACKBONE

        self.num_classes = Config.NUM_CLASSES
        self.embedding_dim = Config.EMBEDDING_DIM

        # 1. Backbone
        # num_classes=0 and global_pool="" ensures we get raw feature maps
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Determine the number of input features from the backbone
        # Run a dummy forward pass to get shapes dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, Config.IMG_SIZE, Config.IMG_SIZE)
            features = self.backbone(dummy_input)
            self.in_features = features.shape[1]

        # 2. Pooling
        self.pooling = nn.AdaptiveAvgPool2d(1)

        # 3. Projection Neck
        # BatchNorm -> Dropout -> Linear -> BatchNorm
        self.neck = nn.Sequential(
            nn.BatchNorm1d(self.in_features),
            nn.Dropout(p=Config.DROPOUT),
            nn.Linear(self.in_features, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
        )

        # 4. ArcFace Head
        self.head = ArcMarginProduct(
            in_features=self.embedding_dim,
            out_features=self.num_classes,
            s=Config.ARC_SCALE,
            m=Config.ARC_MARGIN,
            easy_margin=False,
        )

    def forward(self, x, labels=None):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input images (B, C, H, W)
            labels (torch.Tensor, optional): Ground truth labels (B,).
                                             If None, returns inference logits.

        Returns:
            torch.Tensor: Logits (B, num_classes)
        """
        # Feature Extraction
        x = self.backbone(x)  # (B, C, H, W)
        x = self.pooling(x)  # (B, C, 1, 1)
        x = x.view(x.size(0), -1)  # (B, C)

        # Projection
        embeddings = self.neck(x)  # (B, embedding_dim)

        # Head (Metric Learning Loss / Inference)
        logits = self.head(embeddings, labels)

        return logits
