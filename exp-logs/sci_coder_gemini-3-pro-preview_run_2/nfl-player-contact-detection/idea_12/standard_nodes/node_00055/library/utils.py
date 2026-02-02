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
    Computes the Matthews Correlation Coefficient.

    Args:
        y_true: Ground truth labels (Tensor or numpy array).
        y_pred: Predicted binary labels (Tensor or numpy array).

    Returns:
        float: The MCC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are integers (binary)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    return matthews_corrcoef(y_true, y_pred)


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with logits.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Where p_t is the model's estimated probability for the class with label y=1.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        """
        Args:
            alpha (float): Weighting factor for the rare class (1).
                           If alpha=0.25, class 1 is weighted by 0.25 and class 0 by 0.75.
                           Note: In some implementations, alpha is the weight for class 1.
                           Here we follow the definition: alpha * loss_pos + (1-alpha) * loss_neg.
            gamma (float): Focusing parameter. Higher values focus more on hard examples.
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits output from the model (batch_size, 1) or (batch_size,).
            targets: Ground truth labels (same shape as inputs), float or int.
        """
        # Ensure targets are float for BCE calculation
        targets = targets.float()

        # Calculate standard BCE with logits
        # This computes -log(p_t)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Calculate p_t (probability of the true class)
        # p_t = exp(-bce_loss)
        pt = torch.exp(-bce_loss)

        # Calculate alpha_t
        # If target=1, alpha_t = alpha
        # If target=0, alpha_t = 1 - alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Calculate Focal Loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
