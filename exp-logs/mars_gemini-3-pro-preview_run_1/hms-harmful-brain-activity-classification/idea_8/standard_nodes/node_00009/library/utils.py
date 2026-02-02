import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

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


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training loops.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def kl_divergence_score(y_true, y_pred, epsilon=1e-15):
    """
    Calculates the Kullback-Leibler Divergence between y_true and y_pred.
    Metric: sum(y_true * log(y_true / y_pred))

    Args:
        y_true (np.array): Ground truth probabilities. Shape (N, C).
        y_pred (np.array): Predicted probabilities. Shape (N, C).
        epsilon (float): Small value to prevent log(0) or division by zero.

    Returns:
        float: The average KL divergence score across the samples.
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Calculate KL Divergence: sum(P * log(P / Q)) = sum(P * log(P) - P * log(Q))
    # We add epsilon to y_true inside log to avoid log(0) where y_true is 0
    # Note: limit of x*log(x) as x->0 is 0, so y_true=0 contributes 0.

    term1 = y_true * np.log(y_true + epsilon)
    term2 = y_true * np.log(y_pred)

    # Sum over classes (axis=1)
    kl_per_sample = np.sum(term1 - term2, axis=1)

    # Return mean over batch
    return np.mean(kl_per_sample)


class KLDivLossWithLogits(nn.Module):
    """
    KL Divergence Loss that accepts logits as input and probabilities as target.
    This is numerically stable as it applies log_softmax internally before
    passing to nn.KLDivLoss.
    """

    def __init__(self, reduction="batchmean"):
        """
        Args:
            reduction (str): Specifies the reduction to apply to the output.
                             'batchmean' is mathematically correct for KL Div.
        """
        super(KLDivLossWithLogits, self).__init__()
        self.reduction = reduction
        self.kl_loss = nn.KLDivLoss(reduction=reduction)

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw output from the model (before softmax). Shape (N, C).
            targets (torch.Tensor): Ground truth probabilities. Shape (N, C).

        Returns:
            torch.Tensor: Calculated loss.
        """
        # nn.KLDivLoss expects input to be log-probabilities
        log_probs = F.log_softmax(logits, dim=1)

        # Calculate loss
        # Note: nn.KLDivLoss(log_probs, targets) computes sum(targets * (log(targets) - log_probs))
        # if log_target=False (default). This matches KL Divergence definition.
        loss = self.kl_loss(log_probs, targets)

        return loss
