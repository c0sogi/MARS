import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef
from library.config import Config


def set_seed(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for binary classification.
    Wraps BCEWithLogitsLoss to handle numerical stability with logits.

    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        gamma: float = Config.FOCAL_LOSS_GAMMA,
        alpha: float = None,
        reduction: str = "mean",
    ):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits from the model (batch_size, 1) or (batch_size,)
            targets: Binary targets (0 or 1) with same shape as inputs
        """
        # Ensure targets are float for BCE
        targets = targets.float()

        # Calculate standard BCE with logits
        # reduction='none' so we can apply the focal modulation per element
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Get the probabilities p_t corresponding to the true class
        # p_t = p if y=1 else 1-p
        # With logits: p = sigmoid(x)
        pt = torch.exp(-bce_loss)

        # Focal modulation: (1 - p_t)^gamma
        focal_term = (1 - pt) ** self.gamma

        # Apply alpha weighting if provided
        if self.alpha is not None:
            # alpha_t = alpha if y=1 else (1-alpha)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * focal_term * bce_loss
        else:
            loss = focal_term * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def optimize_mcc_threshold(
    y_true: np.ndarray, y_pred_proba: np.ndarray, num_thresholds: int = 100
):
    """
    Finds the decision threshold that maximizes the Matthews Correlation Coefficient.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred_proba: Predicted probabilities (0 to 1).
        num_thresholds: Number of steps in the grid search.

    Returns:
        best_threshold (float): The threshold value giving the highest MCC.
        best_score (float): The highest MCC score achieved.
    """
    best_threshold = 0.5
    best_score = -1.0

    # Generate thresholds to test (avoiding 0 and 1 exactly to prevent edge case errors)
    thresholds = np.linspace(0.01, 0.99, num_thresholds)

    for thresh in thresholds:
        # Binarize predictions
        y_pred_bin = (y_pred_proba >= thresh).astype(int)

        # Calculate MCC
        score = matthews_corrcoef(y_true, y_pred_bin)

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score
