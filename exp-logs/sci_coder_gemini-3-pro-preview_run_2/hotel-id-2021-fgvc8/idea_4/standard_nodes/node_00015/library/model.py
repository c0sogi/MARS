import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
from library.config import CFG


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


class ArcFace(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss.
    Used for the Coarse-Grained Chain ID Head.
    """

    def __init__(
        self, in_features, out_features, scale=30.0, margin=0.50, easy_margin=False
    ):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):
        # --------------------------- cosine ---------------------------
        # Normalize input and weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # --------------------------- margin ---------------------------
        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # The above is equivalent to the following efficient implementation:

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        output *= self.scale
        return output


class SubCenterArcFace(nn.Module):
    """
    Sub-Center ArcFace.
    Used for the Fine-Grained Hotel ID Head to handle intra-class variance.
    """

    def __init__(
        self, in_features, out_features, k=3, scale=30.0, margin=0.50, easy_margin=False
    ):
        super(SubCenterArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        # Weights shape: (out_features * k, in_features)
        self.weight = nn.Parameter(torch.FloatTensor(out_features * k, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):
        # --------------------------- cosine ---------------------------
        # Normalize input and weights
        # input: (Batch, Dim)
        # weight: (Classes*K, Dim)
        # cosine: (Batch, Classes*K)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # Reshape to (Batch, Classes, K)
        cosine = cosine.view(-1, self.out_features, self.k)

        # Max-pool over K to find the best sub-center for each class
        cosine, _ = torch.max(cosine, dim=2)

        # --------------------------- margin ---------------------------
        # Standard ArcFace logic follows on the max-pooled cosine
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        output *= self.scale
        return output


class HotelNet(nn.Module):
    """
    Hierarchical Multi-Task Network for Hotel Identification.
    Backbone: EfficientNet-B5
    Pooling: GeM
    Heads: SubCenterArcFace (Hotel), ArcFace (Chain)
    """

    def __init__(self):
        super(HotelNet, self).__init__()

        # 1. Backbone
        self.backbone = timm.create_model(CFG.backbone, pretrained=CFG.pretrained)

        # Determine feature dimension
        if hasattr(self.backbone, "classifier"):
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, "fc"):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, "head"):
            in_features = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()
        else:
            # Fallback for some timm models, though efficientnet usually has classifier
            in_features = self.backbone.num_features

        self.backbone.global_pool = nn.Identity()

        # 2. Pooling & Neck
        self.pooling = GeM()
        self.bn1 = nn.BatchNorm1d(in_features)
        self.fc = nn.Linear(in_features, CFG.embedding_size)
        self.bn2 = nn.BatchNorm1d(CFG.embedding_size)

        # 3. Heads
        # Fine-Grained Head (Hotel ID)
        self.hotel_head = SubCenterArcFace(
            in_features=CFG.embedding_size,
            out_features=CFG.num_classes,
            k=CFG.subcenter_k,
            scale=CFG.scale_hotel,
            margin=CFG.margin_hotel,
        )

        # Coarse-Grained Head (Chain ID)
        self.chain_head = ArcFace(
            in_features=CFG.embedding_size,
            out_features=CFG.num_chains,
            scale=CFG.scale_chain,
            margin=CFG.margin_chain,
        )

    def extract_features(self, x):
        """
        Extracts embeddings from images.
        """
        # Backbone forward
        x = self.backbone.forward_features(x)

        # Pooling (N, C, H, W) -> (N, C, 1, 1)
        x = self.pooling(x)
        x = x.flatten(1)

        # Neck
        x = self.bn1(x)
        x = self.fc(x)
        x = self.bn2(x)

        return x

    def forward(self, x, hotel_label=None, chain_label=None):
        """
        Forward pass.
        If labels are provided (Training), returns logits for both heads.
        If labels are None (Inference), returns embeddings.
        """
        embedding = self.extract_features(x)

        if self.training and hotel_label is not None and chain_label is not None:
            hotel_logits = self.hotel_head(embedding, hotel_label)
            chain_logits = self.chain_head(embedding, chain_label)
            return hotel_logits, chain_logits
        else:
            return embedding
