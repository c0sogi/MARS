import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ArcFaceLoss(nn.Module):
    """
    ArcFace (Additive Angular Margin Loss) Module.

    This module implements the ArcFace loss function, which adds an additive angular margin
    to the target classification logits to enforce higher intra-class compactness and
    inter-class discrepancy. It maintains its own learnable weights (class prototypes).

    Reference: Deng et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition".
    """

    def __init__(self, in_features, out_features, s=None, m=None):
        """
        Initialize the ArcFaceLoss module.

        Args:
            in_features (int): Dimension of the input embeddings.
            out_features (int): Number of classes (identities).
            s (float, optional): Norm of input feature. Defaults to Config.ARCFACE_S.
            m (float, optional): Margin. Defaults to Config.ARCFACE_M.
        """
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Load defaults from Config if not provided
        self.s = s if s is not None else Config.ARCFACE_S
        self.m = m if m is not None else Config.ARCFACE_M

        # Learnable weights (prototypes) for each class
        # Shape: (out_features, in_features)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Epsilon for numerical stability in acos
        self.eps = 1e-7

    def forward(self, embeddings, labels):
        """
        Forward pass of the ArcFace Loss.

        Args:
            embeddings (torch.Tensor): Input embeddings of shape (batch_size, in_features).
            labels (torch.Tensor): Ground truth labels of shape (batch_size,).

        Returns:
            torch.Tensor: Scalar CrossEntropy loss value.
        """
        # 1. Normalize inputs (embeddings) and weights (prototypes)
        # L2 Normalization ensures the dot product equals cosine similarity
        norm_embeddings = F.normalize(embeddings, p=2, dim=1)
        norm_weight = F.normalize(self.weight, p=2, dim=1)

        # 2. Calculate Cosine Similarity (Logits)
        # Shape: (batch_size, out_features)
        cosine = F.linear(norm_embeddings, norm_weight)

        # 3. Angular Margin Penalty
        # Clamp cosine values to avoid NaN in acos (numerical stability)
        cosine_clamped = torch.clamp(cosine, -1.0 + self.eps, 1.0 - self.eps)

        # Calculate theta = arccos(cosine)
        theta = torch.acos(cosine_clamped)

        # Add margin only to the target class
        # theta_target = theta + m
        theta_m = theta + self.m

        # Calculate new logits: cos(theta + m)
        logits_m = torch.cos(theta_m)

        # 4. Construct Final Logits
        # Create a one-hot mask for the ground truth labels
        # labels need to be LongTensor for scatter
        labels = labels.long()
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # Where the mask is 1 (target class), use logits_m (margin applied)
        # Where the mask is 0 (other classes), use original cosine
        logits = torch.where(one_hot == 1.0, logits_m, cosine)

        # 5. Rescale Logits
        # Multiply by the scale factor s
        logits = logits * self.s

        # 6. Compute Cross Entropy Loss
        loss = F.cross_entropy(logits, labels)

        return loss
