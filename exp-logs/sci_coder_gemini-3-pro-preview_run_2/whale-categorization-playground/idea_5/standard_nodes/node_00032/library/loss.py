import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import Config


class ArcFaceLoss(nn.Module):
    """
    ArcFace (Additive Angular Margin Loss) Module.

    This module acts as the classification head and loss function. It maintains
    the learnable weights (class centers), computes the cosine similarity between
    embeddings and weights, applies the angular margin penalty to the ground truth
    classes, scales the result, and computes the Cross Entropy loss.
    """

    def __init__(self, num_classes, embedding_size, s=Config.arc_s, m=Config.arc_m):
        """
        Initialize the ArcFaceLoss module.

        Args:
            num_classes (int): The number of classes (identities) in the training set.
            embedding_size (int): The dimension of the input embeddings.
            s (float, optional): The scaling factor (inverse temperature). Defaults to Config.arc_s.
            m (float, optional): The angular margin penalty. Defaults to Config.arc_m.
        """
        super(ArcFaceLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.s = s
        self.m = m

        # Learnable weights (Class Centers)
        # Shape: (num_classes, embedding_size)
        # These are the "W" in the ArcFace paper.
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))

        # Initialize weights
        nn.init.xavier_uniform_(self.weight)

        # Precompute constants for the forward pass
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # Threshold for numerical stability and handling angles where theta + m > pi
        # cos(pi - m) = -cos(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        """
        Forward pass to compute the ArcFace loss.

        Args:
            embeddings (torch.Tensor): Input features/embeddings. Shape: (batch_size, embedding_size).
            labels (torch.Tensor): Ground truth class indices. Shape: (batch_size,).

        Returns:
            torch.Tensor: The computed scalar Cross Entropy loss.
        """
        # 1. Normalize Inputs
        # Normalize weights and embeddings to lie on the hypersphere
        # F.normalize defaults to L2 norm (p=2) along dim=1
        norm_weights = F.normalize(self.weight, p=2, dim=1)
        norm_embeddings = F.normalize(embeddings, p=2, dim=1)

        # 2. Compute Cosine Similarity
        # Output shape: (batch_size, num_classes)
        # equivalent to matmul(norm_embeddings, norm_weights.T)
        cosine = F.linear(norm_embeddings, norm_weights)

        # 3. Numerical Stability
        # Clamp cosine values to range [-1+eps, 1-eps] to prevent NaN in sqrt/acos
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        # 4. Apply Angular Margin
        # We want to compute cos(theta + m)
        # Formula: cos(a + b) = cos(a)cos(b) - sin(a)sin(b)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        # Handle cases where theta + m > pi (where cosine function is not monotonic)
        # If cos(theta) > cos(pi - m), then theta < pi - m, so theta + m < pi.
        # In this safe region, we use phi.
        # Otherwise, we use a fallback (cosine - mm) to ensure the penalty is applied correctly.
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 5. Create Target Logits
        # We only apply the margin penalty to the ground truth class logits
        # Create one-hot encoding of labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Combine: Use phi for target class, cosine for others
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # 6. Scale Logits
        output *= self.s

        # 7. Compute Loss
        loss = F.cross_entropy(output, labels)

        return loss
