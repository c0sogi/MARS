import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class FocalLoss(nn.Module):
    """
    Binary Focal Loss implementation for addressing class imbalance.

    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha (float): Balancing factor. For binary classification, alpha applies to class 1,
                       and (1-alpha) applies to class 0.
        gamma (float): Focusing parameter to suppress easy examples.
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
            inputs (Tensor): Logits from the model (before sigmoid). Shape (N, *)
            targets (Tensor): Ground truth labels (0 or 1). Shape (N, *)
        """
        # Compute standard BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Get the probabilities (p_t) associated with the ground truth class
        # pt = exp(-bce_loss) is a numerically stable way to get p_t
        pt = torch.exp(-bce_loss)

        # Determine alpha factor for each sample
        # If target=1, factor=alpha; if target=0, factor=1-alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Compute Focal Loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def optimize_threshold(y_true, y_pred_prob, steps=Config.THRESHOLD_SEARCH_STEPS):
    """
    Performs a grid search to find the decision threshold that maximizes the
    Matthews Correlation Coefficient (MCC).

    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred_prob (array-like): Predicted probabilities (0 to 1).
        steps (int): Number of threshold steps to evaluate.

    Returns:
        tuple: (best_threshold, best_mcc)
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)

    thresholds = np.linspace(0, 1, steps)
    best_mcc = -1.0
    best_threshold = 0.5

    for thresh in thresholds:
        # Convert probabilities to binary predictions based on current threshold
        y_pred = (y_pred_prob >= thresh).astype(int)

        # Calculate MCC
        # Note: MCC can handle cases where predictions are all 0 or all 1 gracefully (returns 0)
        mcc = matthews_corrcoef(y_true, y_pred)

        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = thresh

    return best_threshold, best_mcc
