import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class StableFocalLoss(nn.Module):
    """
    Implements the Focal Loss for binary classification with explicit float32 casting
    to ensure numerical stability during Mixed Precision training.

    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Where p_t is the model's estimated probability for the true class.
    """

    def __init__(
        self, alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA, reduction="mean"
    ):
        """
        Args:
            alpha (float): Weighting factor for the rare class (1).
                           If alpha=0.25, class 1 weight is 0.25, class 0 weight is 0.75.
            gamma (float): Focusing parameter to down-weight easy examples.
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super(StableFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Logits from the model (before sigmoid). Shape (N, *)
            targets (torch.Tensor): Binary ground truth labels. Shape (N, *)

        Returns:
            torch.Tensor: Computed loss.
        """
        # Ensure inputs and targets are float32 to prevent NaNs in AMP
        inputs = inputs.float()
        targets = targets.float()

        # Flatten to ensure shapes match (N, 1)
        inputs = inputs.view(-1, 1)
        targets = targets.view(-1, 1)

        # Compute standard BCE with logits (numerically stable)
        # reduction='none' is required to apply focal weights element-wise
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Calculate p_t (probability of the true class)
        # Since BCE = -log(p_t), we can compute p_t = exp(-BCE)
        pt = torch.exp(-bce_loss)

        # Compute the focal term: (1 - p_t)^gamma
        focal_term = (1 - pt) ** self.gamma

        # Compute the alpha weighting term
        if self.alpha is not None:
            # alpha for class 1, (1 - alpha) for class 0
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * focal_term * bce_loss
        else:
            loss = focal_term * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
