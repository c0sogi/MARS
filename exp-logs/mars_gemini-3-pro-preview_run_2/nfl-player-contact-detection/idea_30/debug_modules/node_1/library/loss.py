import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class FocalLoss(nn.Module):
    """
    Implements Focal Loss for binary classification with logits.

    Focal Loss addresses class imbalance by down-weighting well-classified examples
    and focusing training on hard negatives. It is defined as:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t is the model's estimated probability for the true class.

    Attributes:
        alpha (float): Weighting factor for the positive class (1).
                       The negative class (0) will be weighted by (1 - alpha).
        gamma (float): Focusing parameter. Higher values reduce the loss contribution
                       of easy examples.
        reduction (str): Specifies the reduction to apply to the output:
                         'none' | 'mean' | 'sum'.
    """

    def __init__(self, alpha=None, gamma=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        # Use defaults from Config if not provided
        self.alpha = alpha if alpha is not None else Config.FOCAL_LOSS_ALPHA
        self.gamma = gamma if gamma is not None else Config.FOCAL_LOSS_GAMMA
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Logits from the model. Shape (N, *)
            targets (torch.Tensor): Ground truth labels (0 or 1). Shape (N, *)

        Returns:
            torch.Tensor: The calculated loss.
        """
        # Ensure targets are float for BCE calculation
        targets = targets.float()

        # Ensure inputs and targets have the same shape
        if inputs.shape != targets.shape:
            inputs = inputs.view_as(targets)

        # Calculate Binary Cross Entropy with Logits
        # reduction='none' is essential to apply element-wise modulation (focal term)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Calculate p_t (probability of the true class)
        # Since BCE = -log(p_t), we can compute p_t = exp(-BCE)
        pt = torch.exp(-bce_loss)

        # Calculate the Focal term: (1 - p_t)^gamma
        focal_term = (1 - pt) ** self.gamma

        # Calculate the final element-wise loss
        loss = focal_term * bce_loss

        # Apply Alpha balancing if alpha is set
        if self.alpha is not None:
            # alpha_t = alpha if target=1 else (1-alpha)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
