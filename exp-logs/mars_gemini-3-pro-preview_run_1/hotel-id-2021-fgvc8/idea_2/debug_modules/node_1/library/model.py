import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
from library.utils import Config


class ArcFaceHead(nn.Module):
    """
    ArcFace (Additive Angular Margin Loss) Head.

    Reference: Deng et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition".
    """

    def __init__(self, in_features, out_features, scale=30.0, margin=0.50):
        super(ArcFaceHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin

        # Class centers (weights)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precompute constants for margin calculation
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

        # Thresholds for numerical stability
        # theta < pi - m  =>  cos(theta) > cos(pi - m)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label=None):
        """
        Args:
            input: Feature embeddings (Batch, Embedding_Dim)
            label: Ground truth labels (Batch,). If None, returns raw cosine similarities.
        """
        # 1. Normalize inputs and weights to hypersphere
        # input: (B, D) -> (B, D)
        x_norm = F.normalize(input, p=2, dim=1)
        # weight: (C, D) -> (C, D)
        w_norm = F.normalize(self.weight, p=2, dim=1)

        # 2. Compute Cosine Similarity
        # cosine: (B, C)
        cosine = F.linear(x_norm, w_norm)

        # 3. If inference, return scaled cosine similarities
        if label is None:
            return cosine * self.scale

        # 4. If training, apply Additive Angular Margin
        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        # Handle numerical stability (keep phi only where cos(theta) > th)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Create one-hot encoding for targets
        # label must be long tensor
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Apply margin only to target class logits
        # output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # 5. Scale the logits
        output *= self.scale

        return output


class LightweightMetricModel(nn.Module):
    """
    MobileNetV3-based model with a projection neck and ArcFace head.
    """

    def __init__(
        self,
        num_classes=Config.NUM_CLASSES,
        embedding_dim=Config.EMBEDDING_DIM,
        backbone_name=Config.BACKBONE,
        pretrained=True,
    ):
        super(LightweightMetricModel, self).__init__()

        # Handle backbone name mapping (Config uses torchvision style, timm uses specific names)
        if backbone_name == "mobilenet_v3_large":
            timm_backbone = "mobilenetv3_large_100"
        else:
            timm_backbone = backbone_name

        # 1. Backbone
        # num_classes=0 removes the classification head and returns pooled features
        self.backbone = timm.create_model(
            timm_backbone, pretrained=pretrained, num_classes=0
        )

        # Determine backbone output dimension dynamically
        # Cite {debug_lesson_1}: Verify code persistence/state by checking actual values (here, actual output shape).
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, Config.IMG_SIZE, Config.IMG_SIZE)
            dummy_out = self.backbone(dummy_input)
            self.backbone_dim = dummy_out.shape[1]

        # 2. Neck: Embedding Projection
        # Linear -> BatchNorm -> PReLU
        self.neck = nn.Sequential(
            nn.Linear(self.backbone_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.PReLU(),
        )

        # 3. Head: ArcFace
        self.head = ArcFaceHead(
            in_features=embedding_dim,
            out_features=num_classes,
            scale=Config.SCALE,
            margin=Config.MARGIN,
        )

    def forward(self, x, label=None):
        """
        Forward pass.

        Args:
            x: Input images (Batch, C, H, W)
            label: Target labels (Batch,). Optional.

        Returns:
            logits: Scaled cosine similarities (Batch, Num_Classes).
                    If label is provided, target logits have margin penalty applied.
        """
        # Extract features from backbone
        features = self.backbone(x)

        # Project features to embedding space
        embeddings = self.neck(features)

        # Compute logits via ArcFace head
        logits = self.head(embeddings, label)

        return logits

    def extract_features(self, x):
        """
        Extracts normalized embeddings for inference/retrieval.
        """
        features = self.backbone(x)
        embeddings = self.neck(features)
        return F.normalize(embeddings, p=2, dim=1)
