import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def kl_divergence_score(y_true, y_pred, epsilon=1e-15):
    """
    Calculates the Kullback-Leibler (KL) Divergence between the ground truth
    probabilities and the predicted probabilities.

    Metric = sum(y_true * log(y_true / y_pred))

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth probabilities. Shape (N, C).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities. Shape (N, C).
        epsilon (float): Small constant to prevent division by zero or log of zero.

    Returns:
        float: The average KL divergence score.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are float64 for precision
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    # Clip predictions to avoid log(0) and division by zero
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Calculate KL Divergence
    # Formula: sum(P(x) * log(P(x) / Q(x)))
    # Decomposition: sum(P(x) * log(P(x)) - P(x) * log(Q(x)))

    # Term 1: y_true * log(y_true)
    # We use np.where to handle the case where y_true is 0 (0 * log(0) = 0)
    term1 = np.where(y_true > 0, y_true * np.log(y_true), 0.0)

    # Term 2: y_true * log(y_pred)
    term2 = y_true * np.log(y_pred)

    # KL = term1 - term2
    # Sum over classes (axis=1)
    kl_per_sample = np.sum(term1 - term2, axis=1)

    # Return mean over samples
    return np.mean(kl_per_sample)
