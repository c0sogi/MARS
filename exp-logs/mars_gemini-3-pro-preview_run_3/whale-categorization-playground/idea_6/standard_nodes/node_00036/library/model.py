import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class ArcFaceHead(nn.Module):
    """
    ArcFace (Additive Angular Margin Loss) Head.
    Computes the cosine similarity between embeddings and class centers,
    adds a margin to the ground truth class angle, and rescales.
    """

    def __init__(self, in_features, out_features, scale=30.0, margin=0.50):
        super(ArcFaceHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin

        # Weight shape: (out_features, in_features)
        # These represent the class centers on the hypersphere
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Pre-compute constants for the margin function
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

        # Threshold for numerical stability
        # We need to ensure theta + m < pi to maintain monotonicity of cosine
        # If cos(theta) > cos(pi - m), then theta < pi - m
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):
        """
        Args:
            input: (batch_size, in_features) - Normalized embeddings
            label: (batch_size) - Ground truth labels
        """
        # 1. Normalize inputs and weights to place them on the hypersphere
        # F.normalize defaults to p=2, dim=1
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # 2. Calculate cos(theta + m)
        # Identity: cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        # 3. Handle numerical stability / easy margin
        if self.training:
            # If the angle is too large, the margin might push it beyond pi (where cosine increases).
            # In those cases, we use a fallback penalty (cosine - mm).
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

            # 4. Create one-hot encoding to apply margin only to ground truth
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)

            # 5. Apply margin to target class, keep others as is
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

            # 6. Scale the logits
            output *= self.scale
        else:
            # During validation/inference (if not using embeddings directly),
            # we just scale the cosine similarity.
            output = cosine * self.scale

        return output


class WhaleEfficientNetArcFace(nn.Module):
    """
    EfficientNet-B2 backbone with a specific Neck and ArcFace Head.

    Architecture:
      Input -> EfficientNet-B2 (GAP) -> BN -> Dropout -> Linear -> BN -> ArcFace
    """

    def __init__(self, model_name=Config.MODEL_NAME, num_classes=None, pretrained=True):
        super(WhaleEfficientNetArcFace, self).__init__()

        # 1. Backbone
        # num_classes=0 removes the classifier
        # global_pool='avg' ensures we get the pooled feature vector directly
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # Get input feature dimension (e.g., 1408 for EfficientNet-B2)
        in_features = self.backbone.num_features

        # 2. Neck
        # Structure: BatchNorm -> Dropout -> Linear -> BatchNorm
        # We explicitly exclude non-linear activations (ReLU/PReLU) in the neck
        # to preserve the manifold structure for metric learning.
        self.neck = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(in_features, Config.EMBEDDING_SIZE, bias=False),
            nn.BatchNorm1d(Config.EMBEDDING_SIZE),
        )

        # 3. Head
        # Only initialized if num_classes is provided (Training mode)
        if num_classes is not None:
            self.head = ArcFaceHead(
                in_features=Config.EMBEDDING_SIZE,
                out_features=num_classes,
                scale=Config.ARC_SCALE,
                margin=Config.ARC_MARGIN,
            )
        else:
            self.head = None

    def forward(self, images, targets=None):
        """
        Forward pass.

        Args:
            images (torch.Tensor): Input images (Batch, C, H, W)
            targets (torch.Tensor, optional): Class labels. Defaults to None.

        Returns:
            torch.Tensor:
                - If training and targets provided: ArcFace Logits.
                - If inference/val: Embeddings (output of the neck).
        """
        # Feature extraction
        features = self.backbone(images)  # (Batch, Backbone_Dim)

        # Projection
        embeddings = self.neck(features)  # (Batch, Embedding_Size)

        if self.training and targets is not None and self.head is not None:
            logits = self.head(embeddings, targets)
            return logits
        else:
            # Inference or Validation (Embedding retrieval)
            return embeddings
