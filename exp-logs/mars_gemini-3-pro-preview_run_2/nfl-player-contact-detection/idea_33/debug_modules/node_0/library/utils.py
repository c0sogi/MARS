import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef
from library import config


def seed_everything(seed=config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_shortest_arc(angle1, angle2):
    """
    Calculates the shortest arc between two angles in degrees.
    Handles wrapping around 360 degrees.
    Supports scalars or numpy arrays.
    """
    diff = np.abs(angle1 - angle2) % 360
    return np.minimum(diff, 360 - diff)


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification to address class imbalance and easy negatives.
    Formula: Loss = - alpha_t * (1 - pt)^gamma * log(pt)
    """

    def __init__(
        self,
        alpha=config.FOCAL_LOSS_ALPHA,
        gamma=config.FOCAL_LOSS_GAMMA,
        reduction="mean",
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: Logits from the model (before sigmoid)
        targets: Binary ground truth labels (0 or 1)
        """
        # Calculate binary cross entropy with logits
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Calculate pt (probability of the true class)
        # Since bce_loss = -log(pt), pt = exp(-bce_loss)
        pt = torch.exp(-bce_loss)

        # Calculate alpha_t
        # If target=1, alpha_t = alpha
        # If target=0, alpha_t = 1 - alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Apply Focal Loss scaling
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def optimize_threshold(y_true, y_probs):
    """
    Performs a grid search to find the decision threshold that maximizes the
    Matthews Correlation Coefficient (MCC).

    Args:
        y_true: Numpy array of ground truth labels.
        y_probs: Numpy array of predicted probabilities.

    Returns:
        best_thresh: The threshold value achieving the highest MCC.
    """
    best_mcc = -1.0
    best_thresh = 0.5

    # Create search space from config
    thresholds = np.linspace(
        config.THRESHOLD_START, config.THRESHOLD_END, config.THRESHOLD_STEPS
    )

    for thresh in thresholds:
        # Binarize predictions based on current threshold
        y_pred = (y_probs >= thresh).astype(int)

        # Calculate MCC
        mcc = matthews_corrcoef(y_true, y_pred)

        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = thresh

    print(f"Best MCC: {best_mcc}")
    return best_thresh
