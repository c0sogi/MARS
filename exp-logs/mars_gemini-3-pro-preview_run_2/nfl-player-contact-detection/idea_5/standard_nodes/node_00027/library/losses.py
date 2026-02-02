import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification tasks with extreme class imbalance.

    The loss is defined as:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where:
        p_t is the model's estimated probability for the class with label y.
        alpha_t is the weighting factor for the class with label y.
    """

    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean"):
        """
        Args:
            alpha (float): Weighting factor for the positive class (1).
                           The negative class (0) will be weighted by (1 - alpha).
                           Default: 0.75 (penalize false negatives more).
            gamma (float): Focusing parameter. Higher values focus more on hard examples.
                           Default: 2.0.
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. Default: 'mean'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predicted logits (before sigmoid) of shape (N, *)
            targets (torch.Tensor): Ground truth labels of shape (N, *).
                                    Values should be 0 or 1.

        Returns:
            torch.Tensor: The calculated loss.
        """
        # Flatten inputs and targets to ensure shapes match and are 1D
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()

        # Compute binary cross entropy with logits
        # reduction='none' preserves the loss per element
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Calculate p_t (probability of the true class)
        # BCE = -log(p_t) => p_t = exp(-BCE)
        pt = torch.exp(-bce_loss)

        # Calculate alpha_t
        # If target=1, alpha_t = alpha
        # If target=0, alpha_t = 1 - alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Calculate Focal Loss
        # FL = alpha_t * (1 - p_t)^gamma * BCE
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
