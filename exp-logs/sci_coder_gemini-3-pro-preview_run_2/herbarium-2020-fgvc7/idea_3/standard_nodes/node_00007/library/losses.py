import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.utils import set_seed


class FocalLoss(nn.Module):
    """
    Focal Loss for Multi-Class Classification.

    This loss function down-weights easy examples and focuses training on hard negatives,
    which is useful for datasets with high class imbalance.

    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha (float, list, np.ndarray, torch.Tensor, optional):
            Weighting factor.
            - If float/int: Applies a constant weight to all classes.
            - If list/array/tensor: Must be of length C (number of classes).
              Weights are applied corresponding to the target class index.
            - If None: No alpha weighting is applied.
            Default: None.
        gamma (float): Focusing parameter. Higher values focus more on hard examples. Default: 2.0.
        reduction (str): Specifies the reduction to apply to the output: 'mean', 'sum', or 'none'. Default: 'mean'.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        # Process alpha argument
        if isinstance(alpha, (list, np.ndarray)):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, (float, int)):
            alpha = torch.tensor([float(alpha)], dtype=torch.float32)

        # Register alpha as a buffer so it moves to device with the module
        if isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predictions (logits) of shape [N, C].
            targets (torch.Tensor): Ground truth labels of shape [N].

        Returns:
            torch.Tensor: The calculated loss.
        """
        # Calculate Cross Entropy Loss without reduction to get per-sample loss
        # inputs are expected to be logits
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # Calculate probabilities pt = exp(-CE)
        pt = torch.exp(-ce_loss)

        # Calculate the focal term: (1 - pt)^gamma
        focal_term = (1 - pt) ** self.gamma

        # Calculate the initial focal loss
        loss = focal_term * ce_loss

        # Apply alpha weighting if configured
        if self.alpha is not None:
            if self.alpha.numel() == 1:
                # Scalar alpha
                loss = self.alpha * loss
            else:
                # Per-class alpha: gather weight for each target
                # self.alpha is shape [C], targets is shape [N]
                alpha_t = self.alpha[targets]
                loss = alpha_t * loss

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
