import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef
import library.config as config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification tasks with class imbalance.

    Implements the loss function: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t is the model's estimated probability for the true class.

    Uses BCEWithLogitsLoss internally for numerical stability.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        """
        Args:
            alpha (float): Weighting factor for the positive class (1).
                           The negative class (0) will be weighted by (1 - alpha).
            gamma (float): Focusing parameter. Higher values reduce the loss contribution
                           from easy examples.
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predicted logits (before sigmoid). Shape (N, 1) or (N,).
            targets (torch.Tensor): Ground truth labels (0 or 1). Shape matches inputs.

        Returns:
            torch.Tensor: The computed loss.
        """
        # Ensure targets are float for BCE calculation
        targets = targets.float()

        # Compute binary cross entropy loss (numerically stable with logits)
        # reduction='none' is required to apply focal weights per sample
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Calculate p_t (probability of the true class)
        # p_t = p if y=1 else 1-p
        # Since BCE = -log(p_t), we can get p_t via exp(-BCE)
        pt = torch.exp(-bce_loss)

        # Calculate alpha_t
        # alpha_t = alpha if y=1 else 1-alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Calculate Focal Loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def optimize_threshold(
    y_true,
    y_pred_probs,
    start=config.THRESHOLD_SEARCH_START,
    end=config.THRESHOLD_SEARCH_END,
    steps=config.THRESHOLD_SEARCH_STEPS,
):
    """
    Finds the decision threshold that maximizes the Matthews Correlation Coefficient (MCC).

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_pred_probs (np.ndarray): Predicted probabilities (0 to 1).
        start (float): Start of the threshold search range.
        end (float): End of the threshold search range.
        steps (int): Number of steps in the search range.

    Returns:
        tuple: (best_threshold, best_mcc_score)
    """
    thresholds = np.linspace(start, end, steps)
    best_mcc = -1.0
    best_thresh = 0.5

    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred_probs = np.asarray(y_pred_probs)

    for thresh in thresholds:
        y_pred = (y_pred_probs >= thresh).astype(int)

        # Calculate MCC
        # Note: MCC is robust to class imbalance
        mcc = matthews_corrcoef(y_true, y_pred)

        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = thresh

    return best_thresh, best_mcc
