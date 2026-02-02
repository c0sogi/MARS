import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
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


class KLDivLossWithLogits(nn.Module):
    """
    Kullback-Leibler Divergence Loss wrapper that accepts logits.
    It applies LogSoftmax to the input logits before passing them to nn.KLDivLoss.
    """

    def __init__(self, reduction="batchmean"):
        super(KLDivLossWithLogits, self).__init__()
        self.reduction = reduction
        self.kl_loss = nn.KLDivLoss(reduction=reduction)

    def forward(self, logits, targets):
        """
        Args:
            logits: Raw output from the model (batch_size, num_classes).
            targets: Target probabilities (batch_size, num_classes).
        """
        input_log_probs = F.log_softmax(logits, dim=1)
        return self.kl_loss(input_log_probs, targets)


def kl_divergence_score(y_true, y_pred, epsilon=1e-15):
    """
    Calculates the KL Divergence metric using NumPy arrays.

    Args:
        y_true: Ground truth probabilities (N, num_classes).
        y_pred: Predicted probabilities (N, num_classes).
        epsilon: Small constant to prevent log(0).

    Returns:
        The average KL divergence score.
    """
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Ensure predictions sum to 1
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)

    # Clip true values for numerical stability in the log term
    y_true_safe = np.clip(y_true, epsilon, 1 - epsilon)

    # KL(P || Q) = sum(P * log(P / Q)) = sum(P * (log P - log Q))
    terms = y_true * (np.log(y_true_safe) - np.log(y_pred))

    return np.mean(np.sum(terms, axis=1))
