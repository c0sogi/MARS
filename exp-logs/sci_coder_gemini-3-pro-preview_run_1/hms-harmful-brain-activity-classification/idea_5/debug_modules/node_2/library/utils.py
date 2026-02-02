import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Safe for multi-GPU

    # Set environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Enforce deterministic algorithms in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value of a metric.
    Useful for tracking loss and scores during training epochs.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the meter with a new value.

        Args:
            val (float): The current value to record.
            n (int): The weight/batch size associated with the value.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return str(self.avg)


def kl_divergence_score(y_true, y_pred, epsilon=1e-15):
    """
    Calculates the Kullback-Leibler (KL) Divergence between the observed target
    and the predicted probability distributions.

    Formula: D_KL(P || Q) = sum(P(x) * log(P(x) / Q(x)))

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth probabilities (P). Shape (N, C).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities (Q). Shape (N, C).
        epsilon (float): Small constant to prevent log(0) errors.

    Returns:
        float: The average KL divergence score across the batch/dataset.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Clip predictions to ensure they are strictly within [epsilon, 1-epsilon]
    # This avoids log(0) which results in -inf
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Compute KL Divergence: sum(P * log(P / Q)) = sum(P * log(P) - P * log(Q))

    # Term 1: y_true * log(y_true)
    # We must handle the case where y_true is 0.
    # Analytically, lim(x->0) x*log(x) = 0.
    # We compute log only where y_true > 0.
    term_true = np.zeros_like(y_true)
    mask = y_true > 0
    term_true[mask] = y_true[mask] * np.log(y_true[mask])

    # Term 2: y_true * log(y_pred)
    # y_pred is already clipped, so log(y_pred) is safe.
    term_pred = y_true * np.log(y_pred)

    # Calculate element-wise KL
    kl_elements = term_true - term_pred

    # Sum over classes (axis=1) to get KL per sample
    kl_per_sample = np.sum(kl_elements, axis=1)

    # Return the mean KL divergence over the batch
    return float(np.mean(kl_per_sample))
