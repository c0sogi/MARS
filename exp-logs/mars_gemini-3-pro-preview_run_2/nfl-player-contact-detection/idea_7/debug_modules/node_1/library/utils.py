import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import matthews_corrcoef
from library.config import set_seed


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification to address class imbalance.
    Formula: -alpha_t * (1 - p_t)**gamma * log(p_t)
    """

    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: Logits (before sigmoid)
        targets: Binary labels (0 or 1)
        """
        # Ensure targets are float and match device of inputs
        if targets.dtype != inputs.dtype:
            targets = targets.type_as(inputs)

        # Compute standard BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # p_t = exp(-bce_loss) since bce_loss = -log(p_t)
        pt = torch.exp(-bce_loss)

        # Determine alpha factor for each sample
        # If target=1, alpha_t = alpha. If target=0, alpha_t = 1-alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Compute Focal Loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def compute_mcc(y_true, y_pred):
    """
    Calculates the Matthews Correlation Coefficient.
    y_true: Ground truth labels (numpy array).
    y_pred: Predicted binary labels (numpy array).
    """
    return matthews_corrcoef(y_true, y_pred)


def optimize_threshold(y_true, y_prob):
    """
    Performs a grid search to find the probability threshold that maximizes MCC.
    y_true: Ground truth labels (numpy array).
    y_prob: Predicted probabilities (numpy array).
    Returns: The optimal threshold float.
    """
    best_threshold = 0.5
    best_mcc = -1.0

    # Grid search from 0.01 to 0.99
    thresholds = np.arange(0.01, 1.00, 0.01)

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)

        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold

    print("Threshold Optimization Results:")
    print("Best Threshold:", best_threshold)
    print("Best MCC:", best_mcc)

    return best_threshold
