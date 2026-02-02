import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class FocalLoss(nn.Module):
    """
    Focal Loss for dense object detection and imbalanced binary classification.

    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Where:
        p_t is the model's estimated probability for the true class.
        alpha_t is the balancing factor for the class.
        gamma is the focusing parameter.
    """

    def __init__(
        self, alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA, reduction="mean"
    ):
        """
        Args:
            alpha (float): Weighting factor for the positive class (0 < alpha < 1).
                           The negative class will be weighted by (1 - alpha).
                           Default comes from Config (0.75).
            gamma (float): Focusing parameter. Higher values focus more on hard examples.
                           Default comes from Config (2.0).
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. 'mean' is default.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Logits output by the model (before sigmoid).
                                   Shape: [batch_size, 1] or [batch_size].
            targets (torch.Tensor): Ground truth binary labels (0 or 1).
                                    Shape: same as inputs.

        Returns:
            torch.Tensor: The computed loss.
        """
        # Ensure inputs and targets are float for BCE calculation
        inputs = inputs.float()
        targets = targets.float()

        # Compute standard Binary Cross Entropy with Logits
        # reduction='none' is required to apply focal weights element-wise
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Calculate p_t: the probability of the target class
        # Since BCE = -log(p_t), we can compute p_t = exp(-BCE)
        pt = torch.exp(-bce_loss)

        # Calculate alpha_t
        # If target == 1, alpha_t = alpha
        # If target == 0, alpha_t = 1 - alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Calculate the Focal Loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
