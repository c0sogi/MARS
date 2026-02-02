import torch
import torch.nn as nn
from library.config import Config


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for multi-label classification.

    ASL optimizes the trade-off between precision and recall by decoupling the
    loss components for positive and negative samples. It down-weights easy
    negatives (which are dominant in multi-label settings) to focus learning
    on hard negatives and positive samples.

    Reference: "Asymmetric Loss For Multi-Label Classification" (ICCV 2021)
    """

    def __init__(
        self,
        gamma_neg=Config.ASL_GAMMA_NEG,
        gamma_pos=Config.ASL_GAMMA_POS,
        clip=Config.ASL_CLIP,
        eps=1e-8,
        reduction="mean",
    ):
        """
        Args:
            gamma_neg (float): Focusing parameter for negative samples (down-weights easy negatives).
            gamma_pos (float): Focusing parameter for positive samples.
            clip (float): Probability margin for shifting negative samples (hard thresholding).
            eps (float): Small constant for numerical stability in logarithms.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): Logits (before sigmoid) of shape (N, C).
            y (torch.Tensor): Ground truth labels of shape (N, C) (0 or 1).

        Returns:
            torch.Tensor: Calculated loss.
        """
        # Explicit casting to float32 is crucial to prevent NaN during mixed-precision training
        # Logits and targets must be in float32 for stable log/pow computations
        x = x.float()
        y = y.float()

        # Calculate probabilities
        xs_pos = torch.sigmoid(x)

        # --- Positive Component ---
        # Standard Focal Loss term for positives: -y * (1-p)^gamma_pos * log(p)
        # We clamp the input to log to avoid log(0)
        loss_pos = (
            y
            * torch.pow(1.0 - xs_pos, self.gamma_pos)
            * torch.log(xs_pos.clamp(min=self.eps))
        )

        # --- Negative Component ---
        # ASL modification: Shifted probability for negatives
        # p_m = max(p - clip, 0)
        # This hard-thresholds easy negatives (where p < clip) to have 0 loss and 0 gradient
        xs_neg = xs_pos
        if self.clip > 0:
            xs_neg = (xs_neg - self.clip).clamp(min=0)

        # Negative term: -(1-y) * (p_m)^gamma_neg * log(1 - p_m)
        loss_neg = (
            (1.0 - y)
            * torch.pow(xs_neg, self.gamma_neg)
            * torch.log((1.0 - xs_neg).clamp(min=self.eps))
        )

        # Combine components
        # Note: The negative signs from the formulas are applied here
        loss = -(loss_pos + loss_neg)

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
