import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ArcFaceLoss(nn.Module):
    """
    Additive Angular Margin Loss (ArcFace) module.

    Reference: Deng et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition".
    CVPR 2019.
    """

    def __init__(
        self,
        num_classes=None,
        embedding_size=None,
        s=None,
        m=None,
        label_smoothing=None,
    ):
        """
        Args:
            num_classes (int): Number of classes (identities). Defaults to Config.NUM_CLASSES.
            embedding_size (int): Size of the input feature vector. Defaults to Config.EMBEDDING_SIZE.
            s (float): Norm scale factor. Defaults to Config.ARCFACE_S.
            m (float): Angular margin. Defaults to Config.ARCFACE_M.
            label_smoothing (float): Label smoothing epsilon. Defaults to Config.LABEL_SMOOTHING.
        """
        super(ArcFaceLoss, self).__init__()

        # Load configuration or use defaults
        self.num_classes = (
            num_classes if num_classes is not None else Config.NUM_CLASSES
        )
        self.embedding_size = (
            embedding_size if embedding_size is not None else Config.EMBEDDING_SIZE
        )
        self.s = s if s is not None else Config.ARCFACE_S
        self.m = m if m is not None else Config.ARCFACE_M
        self.ls_eps = (
            label_smoothing if label_smoothing is not None else Config.LABEL_SMOOTHING
        )

        # Learnable weight matrix [Number of Classes, Embedding Size]
        self.weight = nn.Parameter(
            torch.FloatTensor(self.num_classes, self.embedding_size)
        )
        nn.init.xavier_uniform_(self.weight)

        # Precompute constants for the margin function
        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # Threshold for numerical stability
        # We need to handle the case where theta + m > pi.
        # In this region, cosine is not monotonic decreasing.
        # th = cos(pi - m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

        # Base classification loss
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.ls_eps)

    def forward(self, features, labels):
        """
        Args:
            features (torch.Tensor): Input embeddings of shape [Batch Size, Embedding Size].
            labels (torch.Tensor): Ground truth labels of shape [Batch Size].

        Returns:
            torch.Tensor: Calculated loss value.
        """
        # 1. Normalize Features and Weights
        # x = x / ||x||
        # w = w / ||w||
        features_norm = F.normalize(features, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # 2. Compute Cosine Similarity (Logits)
        # Shape: [Batch Size, Num Classes]
        cosine = F.linear(features_norm, weight_norm)

        # 3. Compute Margin Logits (phi)
        # sin(theta) = sqrt(1 - cos(theta)^2)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

        # cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # 4. Handle Stability (Easy Margin / Monotonicity)
        # If cos(theta) > cos(pi - m), then theta + m < pi, so simple expansion works.
        # Otherwise, use a linear approximation (cosine - mm) to keep gradients well-behaved.
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 5. Apply Margin only to Ground Truth
        # Create one-hot encoding
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # output = (one_hot * phi) + (not_one_hot * cosine)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # 6. Scale Logits
        output *= self.s

        # 7. Compute Cross Entropy Loss
        loss = self.criterion(output, labels)

        return loss
