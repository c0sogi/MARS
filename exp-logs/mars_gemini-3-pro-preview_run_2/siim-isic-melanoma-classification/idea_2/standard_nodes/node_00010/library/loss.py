import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class WeightedBCE(nn.Module):
    """
    Weighted Binary Cross Entropy Loss for imbalanced datasets.
    Wrapper around F.binary_cross_entropy_with_logits that handles reshaping and device placement.
    """

    def __init__(self, pos_weight):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))

    def forward(self, inputs, targets):
        return F.binary_cross_entropy_with_logits(
            inputs.view(-1, 1), targets.view(-1, 1).float(), pos_weight=self.pos_weight
        )


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification tasks with class imbalance.

    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Where:
        p_t is the model's estimated probability for the true class.
        alpha_t is the balancing factor for the true class.
    """

    def __init__(
        self, alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA, reduction="mean"
    ):
        """
        Args:
            alpha (float): Balancing factor. alpha is used for class 1, 1-alpha for class 0.
            gamma (float): Focusing parameter. Higher values focus more on hard examples.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predicted logits (before sigmoid) of shape (N, 1) or (N,).
            targets (torch.Tensor): Ground truth binary labels of shape (N, 1) or (N,).

        Returns:
            torch.Tensor: The calculated loss.
        """
        # Ensure inputs and targets are of the same shape and type
        inputs = inputs.view(-1, 1)
        targets = targets.view(-1, 1).float()

        # Compute binary cross entropy with logits
        # reduction='none' is essential to apply focal weights element-wise
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # p_t = exp(-BCE) since BCE = -log(p_t)
        # This gives the probability of the true class
        pt = torch.exp(-bce_loss)

        # Calculate alpha_t
        # If target=1, alpha_t = alpha
        # If target=0, alpha_t = 1 - alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Calculate the focal loss component
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
