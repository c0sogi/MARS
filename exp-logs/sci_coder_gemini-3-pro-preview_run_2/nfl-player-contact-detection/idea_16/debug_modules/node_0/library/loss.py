import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification tasks with extreme class imbalance.

    This loss function applies a modulating term to the cross entropy loss in order to focus
    learning on hard misclassified examples. It is defined as:

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t is the model's estimated probability for the target class.

    Attributes:
        alpha (float): Weighting factor for the positive class (and 1-alpha for negative).
                       Taken from Config.FOCAL_ALPHA by default.
        gamma (float): Focusing parameter that adjusts the rate at which easy examples are down-weighted.
                       Taken from Config.FOCAL_GAMMA by default.
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
    """

    def __init__(
        self, alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA, reduction="mean"
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass of the Focal Loss.

        Args:
            inputs (torch.Tensor): Raw logits from the model output (no sigmoid applied).
                                   Shape: (batch_size, 1) or (batch_size,).
            targets (torch.Tensor): Ground truth binary labels (0 or 1).
                                    Shape: Same as inputs.

        Returns:
            torch.Tensor: The computed loss value.
        """
        # Ensure targets are float for calculation
        if targets.dtype != inputs.dtype:
            targets = targets.type_as(inputs)

        # Compute binary cross entropy loss element-wise
        # reduction='none' is essential to apply the focal modulation term element-wise
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Calculate p_t (probability of the true class)
        # Since BCE = -log(p_t), we can compute p_t as exp(-BCE)
        # This is numerically stable compared to computing sigmoid(inputs) manually
        pt = torch.exp(-bce_loss)

        # Compute the Focal Loss term: (1 - pt)^gamma * BCE
        focal_term = (1 - pt) ** self.gamma * bce_loss

        # Apply alpha weighting if configured
        if self.alpha is not None:
            # alpha_t = alpha if target=1, else (1-alpha)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * focal_term
        else:
            loss = focal_term

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
