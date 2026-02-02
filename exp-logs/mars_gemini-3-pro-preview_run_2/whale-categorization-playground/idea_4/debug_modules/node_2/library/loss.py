import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from library.config import Config


class ArcFaceLoss(nn.Module):
    """
    Implementation of ArcFace Loss (Additive Angular Margin Loss).

    This module expects the input `logits` to be the cosine similarity between the
    normalized feature vectors and the normalized class centers (weights).
    It applies the angular margin penalty to the target class and scales the result
    before computing Cross Entropy Loss.

    Reference: Deng et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    https://arxiv.org/abs/1801.07698
    """

    def __init__(self, s=Config.SCALE, m=Config.MARGIN, easy_margin=False):
        """
        Args:
            s (float): Norm of input feature (Scale). Default: 30.0
            m (float): Margin value. Default: 0.50
            easy_margin (bool): If True, relaxes the margin constraint for stability
                                when theta is large. Default: False.
        """
        super(ArcFaceLoss, self).__init__()
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        # Precompute constants for the cosine addition formula
        # cos(a + m) = cos(a)cos(m) - sin(a)sin(m)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # Thresholds for stability when theta + m > pi
        # th = cos(pi - m)
        self.th = math.cos(math.pi - m)
        # mm = sin(pi - m) * m  (Used as a fallback penalty)
        self.mm = math.sin(math.pi - m) * m

        self.crit = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        """
        Args:
            logits (torch.Tensor): Cosine similarity matrix of shape (batch_size, num_classes).
                                   Values should be in range [-1, 1].
            labels (torch.Tensor): Ground truth labels of shape (batch_size,).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # 1. Clamp cosine values for numerical stability
        # Prevents NaN in sqrt(1 - cos^2) and acos
        cosine = logits.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # 2. Calculate sin(theta)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # 3. Calculate cos(theta + m)
        # Formula: cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # 4. Handle boundary conditions (theta + m > pi)
        if self.easy_margin:
            # If theta > pi/2 (cosine < 0), adding margin might be unstable.
            # easy_margin=True skips the margin for these hard cases.
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Standard implementation:
            # If cos(theta) > cos(pi - m), we are safe to use phi.
            # Otherwise, we use a Taylor expansion approximation or fixed penalty
            # to ensure the function remains monotonic and differentiable.
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 5. Create one-hot encoding for targets
        # We only apply the margin penalty to the ground truth class.
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # 6. Construct final logits
        # For target class (one_hot=1): use phi (margin applied)
        # For other classes (one_hot=0): use cosine (original)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # 7. Apply Feature Scaling
        output *= self.s

        # 8. Compute Cross Entropy
        loss = self.crit(output, labels)

        return loss
