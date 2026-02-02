import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import matthews_corrcoef
from library.config import Config


class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss for binary classification.
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Attributes:
        alpha (float): Weighting factor for the rare class (foreground).
        gamma (float): Focusing parameter to down-weight easy examples.
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
        Args:
            inputs (torch.Tensor): Logits from the model (before sigmoid). Shape (N, *)
            targets (torch.Tensor): Ground truth binary labels. Shape (N, *)

        Returns:
            torch.Tensor: Computed Focal Loss.
        """
        # Ensure targets are float for BCE calculation
        targets = targets.float()

        # Compute standard binary cross entropy (element-wise)
        # log(p_t) is effectively -bce_loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # p_t = exp(-bce_loss)
        p_t = torch.exp(-bce_loss)

        # Calculate alpha_t
        # alpha_t = alpha if target=1 else (1-alpha)
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        else:
            alpha_t = 1.0

        # Compute Focal Loss
        # loss = alpha_t * (1 - p_t)^gamma * BCE
        loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def calc_mcc(y_true, y_pred):
    """
    Helper wrapper to calculate Matthews Correlation Coefficient.

    Args:
        y_true (np.array): Ground truth binary labels.
        y_pred (np.array): Predicted binary labels.

    Returns:
        float: MCC score.
    """
    return matthews_corrcoef(y_true, y_pred)


def optimize_threshold(y_true, y_pred_probs):
    """
    Performs a grid search to find the decision threshold that maximizes the MCC.

    Args:
        y_true (np.array): Ground truth binary labels.
        y_pred_probs (np.array): Predicted probabilities (0 to 1).

    Returns:
        tuple: (best_threshold, best_mcc_score)
    """
    # Define search space based on Config
    thresholds = np.linspace(0, 1, Config.THRESHOLD_STEPS)

    best_mcc = -1.0
    best_threshold = 0.5

    # Iterate through thresholds to find the optimum
    for thresh in thresholds:
        # Convert probabilities to binary predictions based on current threshold
        y_pred_binary = (y_pred_probs >= thresh).astype(int)

        # Calculate MCC
        current_mcc = matthews_corrcoef(y_true, y_pred_binary)

        if current_mcc > best_mcc:
            best_mcc = current_mcc
            best_threshold = thresh

    return best_threshold, best_mcc
