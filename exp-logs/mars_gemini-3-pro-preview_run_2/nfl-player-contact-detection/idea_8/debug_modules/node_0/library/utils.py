import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef
from library import config


def set_seed(seed=config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_mcc(y_true, y_pred):
    """
    Computes the Matthews Correlation Coefficient between ground truth and predictions.

    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted binary labels.

    Returns:
        float: The MCC score.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return matthews_corrcoef(y_true, y_pred)


class FocalLoss(nn.Module):
    """
    Implementation of Binary Focal Loss to handle class imbalance.

    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self, alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA, reduction="mean"
    ):
        """
        Args:
            alpha (float): Weighting factor for the rare class (contact).
            gamma (float): Focusing parameter to down-weight easy examples.
            reduction (str): Specifies the reduction to apply to the output: 'none', 'mean', or 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Logits from the model (before sigmoid). Shape (N, *)
            targets (torch.Tensor): Ground truth binary labels (0 or 1). Shape same as inputs.

        Returns:
            torch.Tensor: The computed loss.
        """
        # Ensure targets are float for BCE calculation
        targets = targets.float()

        # Flatten inputs and targets to ensure 1D correspondence
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Compute binary cross entropy with logits (provides numerical stability)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Calculate probabilities
        probs = torch.sigmoid(inputs)

        # Calculate p_t: p if y=1, 1-p if y=0
        p_t = torch.where(targets == 1, probs, 1 - probs)

        # Calculate alpha_t: alpha if y=1, 1-alpha if y=0
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # Compute the focal loss component
        loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
