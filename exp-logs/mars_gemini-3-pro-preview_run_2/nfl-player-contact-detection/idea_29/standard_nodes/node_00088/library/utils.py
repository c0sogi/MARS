import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
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
    Binary Focal Loss implementation for addressing class imbalance.

    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha (float): Weighting factor for the positive class (1).
                       The negative class (0) will be weighted by (1 - alpha).
        gamma (float): Focusing parameter. Higher values down-weight easy examples.
        reduction (str): 'mean', 'sum', or 'none'.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw output from the model (before sigmoid). Shape (N, 1) or (N,).
            targets (torch.Tensor): Ground truth labels (0 or 1). Shape same as logits.
        """
        # Ensure targets are float and same shape as logits
        targets = targets.view_as(logits).float()

        # Compute binary cross entropy loss (log(p_t))
        # reduction='none' is needed to apply weighting per sample
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Calculate probabilities: p = sigmoid(logits)
        probs = torch.sigmoid(logits)

        # Calculate p_t: p if y=1 else 1-p
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Calculate alpha_t: alpha if y=1 else 1-alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Calculate Focal Loss
        # FL = alpha_t * (1 - p_t)^gamma * BCE
        loss = alpha_t * (1 - p_t).pow(self.gamma) * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def compute_mcc(y_true, y_pred):
    """
    Calculates the Matthews Correlation Coefficient.

    Args:
        y_true (np.ndarray or list): Ground truth binary labels.
        y_pred (np.ndarray or list): Predicted binary labels.

    Returns:
        float: The MCC score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return matthews_corrcoef(y_true, y_pred)


def optimize_threshold(y_true, y_probs, steps=100):
    """
    Finds the optimal decision threshold that maximizes MCC.

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_probs (np.ndarray): Predicted probabilities (0.0 to 1.0).
        steps (int): Number of threshold steps to search between 0 and 1.

    Returns:
        tuple: (best_threshold, best_mcc_score)
    """
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    best_threshold = 0.5
    best_score = -1.0

    # Generate thresholds avoiding 0.0 and 1.0 exactly to prevent edge case issues
    thresholds = np.linspace(0.01, 0.99, steps)

    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        score = matthews_corrcoef(y_true, y_pred)

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score
