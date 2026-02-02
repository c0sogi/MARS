import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calc_mcc(y_true, y_pred):
    """
    Calculates the Matthews Correlation Coefficient (MCC).

    Args:
        y_true: Ground truth labels (numpy array or torch tensor).
        y_pred: Predicted labels (numpy array or torch tensor).

    Returns:
        float: The MCC score.
    """
    # Handle PyTorch tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are binary integers
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    return matthews_corrcoef(y_true, y_pred)


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification tasks with class imbalance.
    Computes FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t).
    """

    def __init__(self, alpha=None, gamma=None, reduction="mean"):
        """
        Args:
            alpha (float, optional): Weighting factor for the positive class.
                                     Defaults to Config.FOCAL_ALPHA.
            gamma (float, optional): Focusing parameter.
                                     Defaults to Config.FOCAL_GAMMA.
            reduction (str, optional): Specifies the reduction to apply to the output:
                                       'none' | 'mean' | 'sum'. Defaults to 'mean'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha if alpha is not None else Config.FOCAL_ALPHA
        self.gamma = gamma if gamma is not None else Config.FOCAL_GAMMA
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Logits from the model of shape (batch_size, 1) or (batch_size,).
            targets (torch.Tensor): Binary labels of shape (batch_size, 1) or (batch_size,).

        Returns:
            torch.Tensor: The computed loss.
        """
        # Flatten inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()

        # Compute binary cross entropy with logits
        # reduction='none' to apply weights element-wise first
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # p_t is the probability of the true class
        # If target=1, p_t = sigmoid(input)
        # If target=0, p_t = 1 - sigmoid(input)
        # This is mathematically equivalent to exp(-bce_loss)
        pt = torch.exp(-bce_loss)

        # Determine alpha_t
        # If target=1, alpha_t = alpha
        # If target=0, alpha_t = 1 - alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Compute Focal Loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
